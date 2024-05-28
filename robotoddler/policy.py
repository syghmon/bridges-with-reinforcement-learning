import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from robotoddler.utils import *
from assembly_gym.utils.geometry import collision_rectangles
from shapely.geometry import Polygon
#from assembly_gym.utils import assemble_hexagonal_ground, assemble_hexagonal_hexagonal

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads

        # Learnable weight matrices for linear transformations
        self.W_k = nn.Parameter(torch.Tensor(num_heads, in_features, out_features))
        self.W_q = nn.Parameter(torch.Tensor(num_heads, in_features, out_features))
        self.W_v = nn.Parameter(torch.Tensor(num_heads, in_features, out_features))
        self.W_m = nn.Parameter(torch.Tensor(num_heads))
        
        self.g = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.ReLU()
        )

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_v)
        nn.init.ones_(self.W_m)

    def forward(self, x):
        #heads = []
        out = 0
        for i in range(self.num_heads):
            # Linear transformations for keys, queries, and values
            key = torch.matmul(x, self.W_k[i])
            query = torch.matmul(x, self.W_q[i])
            value = torch.matmul(x, self.W_v[i])

            # Compute attention scores
            e = torch.matmul(query, key.transpose(0, 1))
            #attention = F.softmax(e.masked_fill(adj == 0, -1e9), dim=1)
            attention = F.softmax(e, dim=1)
            
            # Apply attention to values
            out += self.W_m[i] * torch.matmul(attention, value)
            #heads.append(h_prime)

        x = x + self.g(x + out) # h(g(h(MHA(x))))
        # Concatenate and combine multi-head attentions
        #multi_head_output = torch.sum(heads, dim=-1)
        return x #multi_head_output



class DuellingGraphAttentionNetwork(nn.Module):
    def __init__(self, block_dim, obstacle_dim, target_dim, hidden_dim, num_rounds, num_heads, n_ground_actions, n_block_actions):
        super(DuellingGraphAttentionNetwork, self).__init__()
        
        self.num_ground_actions = n_ground_actions
        self.num_block_actions = n_block_actions
        self.num_rounds = num_rounds
        
        self.block_encoder = nn.Sequential(
            nn.Linear(block_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.obstacle_encoder = nn.Sequential(
            nn.Linear(obstacle_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.ground_param = nn.Parameter(torch.Tensor(hidden_dim))
        
        self.attention_layer = GraphAttentionLayer(hidden_dim, hidden_dim, num_heads) # nn.ModuleList()
        #for _ in range(num_layers):
        #    self.attention_layers.append(GraphAttentionLayer(hidden_dim, hidden_dim, num_heads))
        
        self.decoder_ground = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_ground_actions)
        )
        self.decoder_blocks = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_block_actions)
        )
        self.decoder_value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.sigma = nn.Sigmoid()

    def forward(self, x): # Do not pass anything for the ground block
        # on-hot encoding at the end: 100 block, 010 obstacle, 001 target
        #x_block = self.block_encoder(x[(x[:, -3] == 1).nonzero().squeeze(1)])
        #x_obstacle = self.obstacle_encoder(x[(x[:, -2] == 1).nonzero().squeeze(1)])
        #x_target = self.target_encoder[(x[:, -1] == 1).nonzero().squeeze(1)]
        x = {key: torch.tensor(x[key]) for key in x.keys()}

        x_ = self.ground_param
        if len(x["blocks"]) > 0:
            x_block = self.block_encoder(x["blocks"])
            x_ = torch.vstack([x_, x_block])
        if len(x["obstacles"]) > 0:
            x_obstacle = self.obstacle_encoder(x["obstacles"])
            x_ = torch.vstack([x_, x_obstacle])
        x_target = self.target_encoder(x["targets"])
        x_ = torch.vstack([x_, x_target])
        num_blocks = len(x_block)
        
        #x = torch.vstack([self.ground_param, x_block, x_obstacle, x_target])
        
        for l in range(self.num_rounds):
            x_ = self.attention_layer(x_)
        
        A_ground = self.decoder_ground(x_[0])
        
        A_blocks = self.decoder_blocks(x_[1:num_blocks+1]) # remove targets, obstacles and ground
        
        value = self.decoder_value(x_.mean(dim=0))

        A_mean = torch.hstack([A_ground, A_blocks.flatten()]).mean()
        
        q_ground = value + A_ground - A_mean
        q_blocks = value + A_blocks - A_mean
        
        q = {"ground":q_ground, "cube":q_blocks}

        return q # q_ground, q_blocks

    
class GraphNetwork(nn.Module):
        def __init__(self, shapes, n_receiving_faces, block_dim, obstacle_dim, target_dim, hidden_dim, num_rounds, num_heads, num_ground_actions, num_block_actions_per_face):
            super(GraphNetwork, self).__init__()
            
            self.num_rounds = num_rounds
            self.shapes = shapes

            block_encoders_list = []
            block_decoders_list = []
            for n_f in n_receiving_faces:
                block_encoders_list.append(nn.Sequential(
                    nn.Linear(block_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                ))
                block_decoders_list.append(nn.Sequential(
                    #nn.Linear(hidden_dim, hidden_dim),
                    #nn.ReLU(),
                    nn.Linear(2*hidden_dim, n_f * num_block_actions_per_face)
                ))
            self.block_encoders = nn.ModuleList(block_encoders_list)
            self.block_decoders = nn.ModuleList(block_decoders_list)
            self.obstacle_encoder = nn.Sequential(
                nn.Linear(obstacle_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.target_encoder = nn.Sequential(
                nn.Linear(target_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            self.ground_param = nn.Parameter(2*torch.rand(hidden_dim)-1) # Uniform in [-1, 1]
            
            self.attention_layer = GraphAttentionLayer(hidden_dim, hidden_dim, num_heads) # nn.ModuleList()
            #for _ in range(num_layers):
            #    self.attention_layers.append(GraphAttentionLayer(hidden_dim, hidden_dim, num_heads))
            
            self.global_decoder = nn.Linear(hidden_dim, hidden_dim)
            #self.value_decoder = nn.Sequential(
            #    nn.Linear(hidden_dim, hidden_dim),
            #    nn.ReLU(),
            #    nn.Linear(hidden_dim, 1)
            #)

            self.ground_decoder = nn.Sequential(
                #nn.Linear(hidden_dim, hidden_dim),
                #nn.ReLU(),
                nn.Linear(2*hidden_dim, num_ground_actions)
            )
            
            self.sigma = nn.Sigmoid()


        def forward(self, x): # Do not pass anything for the ground block
            # assume there is an integer at the end of each block vector x to specify its shape type
            x = {key: torch.tensor(x[key]) for key in x.keys()}
            x_block = torch.Tensor()
            if len(x["blocks"]) > 0:
                for i, e in enumerate(self.block_encoders):
                    if len(x["blocks"][:,-1] == i) > 0:
                        b = x["blocks"][x["blocks"][:,-1] == i][:,:-1] # remove last element indicating its type
                        if len(x_block) == 0:
                            x_block = e(b)
                        else:
                            x_block = torch.vstack([x_block, e(b)])
            x_obstacle = []
            if len(x["obstacles"]) > 0:
                x_obstacle = self.obstacle_encoder(x["obstacles"])
            x_target = self.target_encoder(x["targets"])
            
            x_ = torch.vstack([v for v in [self.ground_param, x_block, x_obstacle, x_target] if len(v) > 0])
            
            for l in range(self.num_rounds):
                x_ = self.attention_layer(x_)

            G = self.global_decoder(x_.mean(axis=0))
            q = {}
            q["ground"] = self.ground_decoder(torch.hstack([x_[0], G]))
            index = 1
            if len(x["blocks"]) > 0:
                for i, (sh, d) in enumerate(zip(self.shapes, self.block_decoders)):
                    n_sh = int(sum(x["blocks"][:,-1] == i))
                    if n_sh > 0:
                        b = x_[index:index+n_sh] # selects appropriate block nodes
                        q[sh.name] = d(torch.hstack([b, G.repeat([n_sh, 1])]))
                        index += n_sh
                    else:
                        q[sh.name] = torch.tensor([-torch.inf])
            
            #value = self.decoder_value(x_.mean(dim=0))
            #q = {}
            #A_ground = self.decoder_ground(x_[0])
            #q["ground"] = value + A_ground - A_ground.mean()

            #num_blocks = len(x["block"])
            #A_blocks = torch.Tensor()
            #for s, d in zip(self.shape, self.blocks_decoder):
            #    b = x_[1:num_blocks+1].where([x["blocks"][-1] == i]) # remove targets, obstacles and ground
            #    A_block = torch.vstack([A_block, d(b)])
            #    q[s.name] =  value + A_blocks - A_blocks.mean(dim=1).view(-1, 1)
            #q_blocks = value + A_blocks - A_blocks.mean(dim=1).view(-1, 1)
            
            return q




class Agent(object):
        def __init__(self, shapes, receiving_faces, coming_faces, block_dim, obstacle_dim, target_dim, hidden_dim, num_layers, num_heads, x_discr_ground, y_discr_ground, x_block_offset, y_block_offset, z_max=torch.inf):
            super(Agent, self).__init__()
            self.x_discr_ground = x_discr_ground
            self.y_discr_ground = y_discr_ground
            self.x_block_offset = x_block_offset
            self.y_block_offset = y_block_offset
            self.z_max = z_max
            self.shapes = shapes # possible shapes that are allowed to manipulate
            #self.n_tot_faces = [] # number of faces on which a block can be placed
            self.receiving_faces = receiving_faces
            self.coming_faces = coming_faces
            self.n_coming_faces = [len(f) for f in self.coming_faces] # number of independent ways to position the shape when PLACED

            n_receiving_faces = [len(f) for f in self.receiving_faces]
            num_ground_actions = len(self.x_discr_ground) * len(self.y_discr_ground) * sum(self.n_coming_faces)
            num_block_actions_per_face = len(self.x_block_offset) * len(self.y_block_offset) * sum(self.n_coming_faces)
            
            #self.policy = DuellingGraphAttentionNetwork(block_dim, obstacle_dim, target_dim, hidden_dim, num_layers, num_heads, num_ground_actions, num_block_actions_per_face)
            self.policy = GraphNetwork(shapes, n_receiving_faces, block_dim, obstacle_dim, target_dim, hidden_dim, num_layers, num_heads, num_ground_actions, num_block_actions_per_face)
        
        def check_collision(self, s, new_block_pos, new_block_shape):
            v_new = list(self.shapes[new_block_shape].vertices()) + np.array([new_block_pos[0], new_block_pos[2]])
            p_new = Polygon(v_new).buffer(-0.001)
            for b in s['blocks']:
                v = list(self.shapes[b[-1]].vertices()) + np.array([b[0], b[2]])
                p = Polygon(v)
                if p.intersects(p_new):
                    return True
            for obs in s['obstacles']: # For small cube obstacles
                v = [[-0.02, -0.02], [0.02, -0.02], [0.02, 0.02], [-0.02, 0.02]] + np.array([obs[0], obs[2]])
                p = Polygon(v)
                if p.intersects(p_new):
                    return True
            return False


        def get_Q_value(self, s, a):
            # target_block_index is the index within blocks OF THE SAME SHAPE
            target_shape, target_block_index, target_block_face, shape, shape_face, offset_x, offset_y = a
            q = self.policy(s)
            if target_shape == -1: # ground. order: placed face, offset y, offset x
                X = len(self.x_discr_ground)
                Y = len(self.y_discr_ground)
                offset_x_index = self.x_discr_ground.index(offset_x)
                offset_y_index = self.y_discr_ground.index(offset_y)
                shape_face_index = self.coming_faces[shape].index(shape_face)
                i = (sum(self.n_coming_faces[:target_shape]) + shape_face_index) * X * Y + offset_y_index * X + offset_x_index
                return q["ground"][i]
            else: # order: target block, target face, placed face, offset y, offset x
                X = len(self.x_block_offset)
                Y = len(self.y_block_offset)
                offset_x_index = self.x_block_offset.index(offset_x)
                offset_y_index = self.y_block_offset.index(offset_y)
                shape_face_index = self.coming_faces[shape].index(shape_face)
                target_block_face_index = self.receiving_faces[shape].index(target_block_face)
                #target_block_index * self.shapes[target_shape].num_faces_2d() * sum(self.n_coming_faces) * X * Y \
                i = target_block_face_index * sum(self.n_coming_faces) * X * Y \
                    + (sum(self.n_coming_faces[:target_shape]) + shape_face_index) * X * Y \
                    + offset_y_index * X + offset_x_index
                return q[self.shapes[target_shape].name][target_block_index][i]

        def select_best_action(self, s, allow_collisions=True):
            # Outputs the action associated with the heighest Q values for state s
            with torch.no_grad():
                q = self.policy(s)

                collision = True
                colliding_actions = []
                while collision:
                    max_qs = torch.tensor([q[key].max() for key in q.keys()])
                    if max_qs.max() == -torch.inf:
                        print("Warning: no available action to the agent !!!")
                        return [-1, -1, -1, 0, 0, 0.5, 0], [] # arbitrary action
                    target_shape = int(max_qs.argmax() - 1)

                    if target_shape == -1:
                        X = len(self.x_discr_ground)
                        Y = len(self.y_discr_ground)
                        target_block_index = -1
                        target_block_face = -1
                        i = int(q["ground"].argmax())
                        shape_face_index = i // (X * Y)
                        shape = 0
                        for n_eff_faces in self.n_coming_faces:
                            if n_eff_faces > shape_face_index:
                                break
                            shape_face_index -= n_eff_faces
                            shape += 1
                        shape_face = self.coming_faces[shape][shape_face_index]
                        offset_y = self.y_discr_ground[(i // X) % Y]
                        offset_x = self.x_discr_ground[i % X]



                        block_pos = [offset_x, offset_y, 0.025]

                        #print("Try action {}".format([target_shape, target_block_index, target_block_face, shape, shape_face, offset_x, offset_y]))
                        #print("Try shape {} at position {} on the floor, ie, i = {} with qval {}".format(shape, block_pos, i, q["ground"][i]))
                        if self.check_collision(s, block_pos, shape):
                            q["ground"][i] = -torch.inf
                            colliding_actions.append([target_shape, target_block_index, target_block_face, shape, shape_face, offset_x, offset_y])
                        else:
                            collision = False
                    else:
                        X = len(self.x_block_offset)
                        Y = len(self.y_block_offset)
                        qs = q[self.shapes[target_shape].name]
                        target_block_index = int(qs.max(dim=1).values.argmax()) # this is the index in the list of same target shape blocks !
                        i = int(qs[target_block_index].argmax())
                        target_block_face_index = i // (sum(self.n_coming_faces) * X * Y)
                        target_block_face = self.receiving_faces[target_shape][target_block_face_index]
                        shape_face_index = i // (X * Y) % sum(self.n_coming_faces)
                        shape = 0
                        for n_eff_faces in self.n_coming_faces:
                            if n_eff_faces > shape_face_index:
                                break
                            shape_face_index -= n_eff_faces
                            shape += 1
                        shape_face = self.coming_faces[shape][shape_face_index]
                        offset_y = self.y_block_offset[(i // X) % Y]
                        offset_x = self.x_block_offset[i % X]

                        # !!! Works when using square blocks only !!!
                        block_pos = [b for b in s['blocks'] if b[-1] == target_shape][target_block_index][:3]
                        x = block_pos[0] + offset_x
                        y = block_pos[1] + offset_y
                        z = block_pos[2] + 0.05 # Not always true for other geometries !!!

                        #print("Try action {}".format([target_shape, target_block_index, target_block_face, shape, shape_face, offset_x, offset_y]))
                        #print("Try shape {} at position {} on block {} with shape {}".format(shape, [x,y,z], target_block_index, target_shape))

                        if z > self.z_max or self.check_collision(s, [x,y,z], shape):
                            q[self.shapes[target_shape].name][target_block_index][i] = -torch.inf
                            colliding_actions.append([target_shape, target_block_index, target_block_face, shape, shape_face, offset_x, offset_y])
                        else:
                            collision = False

                    if allow_collisions:
                        break
                    
                    


                return [target_shape, target_block_index, target_block_face, shape, shape_face, offset_x, offset_y], colliding_actions


# For now, assume only half hexagonal blocks
class Agent_Hexagonal_Blocks(object):

    def __init__(self, block_dim, obstacle_dim, target_dim, hidden_dim, num_layers, num_heads, x_discr_ground, y_discr_ground, num_ground_block_actions, num_block_actions):
        super(Agent_Hexagonal_Blocks, self).__init__()
        self.x_discr_ground = x_discr_ground
        self.y_discr_ground = y_discr_ground
        self.num_ground_block_actions = num_ground_block_actions
        self.num_ground_actions = len(x_discr_ground) * len(y_discr_ground) * num_ground_block_actions
        self.num_block_actions = num_block_actions
        
        self.policy = DuellingGraphAttentionNetwork(block_dim, obstacle_dim, target_dim, hidden_dim, num_layers, num_heads, self.num_ground_actions, self.num_block_actions)
        
    def get_Q_value(self, s, a):
        Q_ground, Q_blocks = self.policy(s)
        if a[0] == 0: # chosen target is ground
            return Q_ground[a[1]]
        else:
            return Q_blocks[a[0]-1, a[1]]
        
    def select_best_index(self, s):
        # Outputs the indices associated witht the heighest Q values for state s
        with torch.no_grad():
            q_ground, q_blocks = self.policy(s)
            
            if len(q_blocks) == 0 or q_ground.max() > q_blocks.max(): # Choose ground as target block
                target_block = 0
                i_max = int(q_ground.argmax())
            else:
                target_block = int(q_blocks.max(dim=1).values.argmax()) + 1
                i_max = int(q_blocks[target_block-1].argmax())
                
            return [target_block, i_max]
    
    def index_to_action(self, s, action):
        # Translates indices into proper actions (pos, orien)
        block_type = 1
        with torch.no_grad():
            block_index = action[0]
            action_index = action[1]
            
            if block_index == 0: # select ground
                x_index = action_index // (len(self.y_discr_ground) * self.num_ground_block_actions)
                y_index = (action_index // self.num_ground_block_actions) % len(self.y_discr_ground)
                action_number = action_index % self.num_ground_block_actions
                x = self.x_discr_ground[x_index]
                y = self.y_discr_ground[y_index]
                pos, orien = assemble_hexagonal_ground(np.array([x,y,0]), action_number)
            else:
                target_pos = np.array(s["blocks"][block_index-1, :3]) #.item() # -1 because the ground is not in the list
                phi = s["blocks"][block_index-1, 3] #.item()
                target_orien = np.array([0, phi, 0])
                pos, orien = assemble_hexagonal_hexagonal(target_pos, target_orien, action_index)
                
            
            return pos, orien, block_type
    



class Agent_Square_Blocks(object):
    def __init__(self, block_dim, obstacle_dim, target_dim, hidden_dim, num_layers, num_heads, x_discr_ground, \
                y_discr_ground, num_x_discr_blocks, num_y_discr_blocks, phi_discr, \
                block_width, block_depth, block_height):
        super(Agent_Square_Blocks, self).__init__()
        self.x_discr_ground = x_discr_ground
        self.y_discr_ground = y_discr_ground
        self.num_x_discr_blocks = num_x_discr_blocks
        self.num_y_discr_blocks = num_y_discr_blocks
        self.phi_discr = phi_discr
        self.num_ground_actions = len(x_discr_ground) * len(y_discr_ground) * len(phi_discr)
        self.num_block_actions = num_x_discr_blocks * num_y_discr_blocks * len(phi_discr)
        self.block_width = block_width # x
        self.block_depth = block_depth # y
        self.block_height = block_height # z

        self.policy = DuellingGraphAttentionNetwork(block_dim, obstacle_dim, target_dim, hidden_dim, num_layers, num_heads, self.num_ground_actions, self.num_block_actions)


    def get_Q_value(self, s, a):
        Q_ground, Q_blocks = self.policy(s)
        if a[0] == 0: # chosen target is ground
            return Q_ground[a[1]]
        else:
            return Q_blocks[a[0]-1, a[1]]
        
    def select_best_index(self, s):
        # Outputs the indices associated witht the heighest Q values for state s
        with torch.no_grad():
            q_ground, q_blocks = self.policy(s)
            
            if len(q_blocks) == 0 or q_ground.max() > q_blocks.max(): # Choose ground as target block
                target_block = 0
                i_max = int(q_ground.argmax())
            else:
                target_block = int(q_blocks.max(dim=1).values.argmax()) + 1
                i_max = int(q_blocks[target_block-1].argmax())
                
            return [target_block, i_max]
    
    def index_to_action(self, s, action):
        # Translates indices into proper actions (pos, orien)
        block_type = 0
        with torch.no_grad():
            block_index = action[0]
            action_index = action[1]
            
            if block_index == 0: # select ground
                x_index = action_index // (len(self.y_discr_ground) * len(self.phi_discr))
                y_index = (action_index // len(self.phi_discr)) % len(self.y_discr_ground)
                phi_index = action_index % len(self.phi_discr)
                x = self.x_discr_ground[x_index]
                y = self.y_discr_ground[y_index]
                phi = self.phi_discr[phi_index]
                pos = np.array([x, y, self.block_height / 2])
                orien = np.array([0, 0, phi])
            else:
                delta_x_index = action_index // (self.num_y_discr_blocks * len(self.phi_discr))
                delta_y_index = (action_index // len(self.phi_discr)) % self.num_y_discr_blocks
                phi_index = action_index % len(self.phi_discr)
                phi = self.phi_discr[phi_index]

                target_pos = np.array(s["blocks"][block_index-1, :3]) # -1 because the ground is not in the list
                target_phi = s["blocks"][block_index-1, 3]

                dx = (-0.45 + 0.9 * delta_x_index / (self.num_x_discr_blocks - 1)) * self.block_width
                if self.num_y_discr_blocks == 1: # for the 2D case
                    dy = 0
                else:
                    dy = (-0.45 + 0.9 * delta_y_index / (self.num_y_discr_blocks - 1)) * self.block_depth
                dx, dy = rotate(dx, dy, target_phi)
                
                pos = target_pos + np.array([dx, dy, self.block_height])
                orien = mod_2pi(np.array([0, 0, phi + target_phi]))                    
            
            return pos, orien, block_type

