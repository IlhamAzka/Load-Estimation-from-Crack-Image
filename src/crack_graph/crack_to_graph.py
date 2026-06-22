#######################################################
"""""""""""
Python program to create crack graph from crack surface
using Shi-Tomasi corner detection algorithm
and pixel tracking algorithm
by: Pedram Bazrafshan
"""""""""""
#######################################################


#######################################################
"""""""""""
Please cite the paper below if you use this code:
Bazrafshan, P., On, T., Basereh, S., Okumus, P., & Ebrahimkhanlou, A. (2024). A graph‐based method for quantifying crack patterns on reinforced concrete shear walls. Computer‐Aided Civil and Infrastructure Engineering, 39(4), 498-517.
"""""""""""
#######################################################


import cv2
import math
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import csv
import networkx as nx

from skimage.morphology import thin
from copy import deepcopy
from .functions import *
from networkx.algorithms import community

# mark: adjacency matrix

def adj_mat(nodes, edges_dict):
    # convert dictionary of edges to connectivity (adjacency) matrix
    adj_matrix = np.zeros((len(nodes), len(nodes)))       # initiate the connectivity matrix with zeros
    for v in edges_dict.values():
        # print("v = ", v)
        # print("eval(v[0]) = ", eval(v[0]))
        adj_matrix[eval(v[0])][eval(v[1])] = v[2]
        adj_matrix[eval(v[1])][eval(v[0])] = v[2]

    return adj_matrix

# mark: arch matching

def arch_match(nodes_dict, img_base, nodes, edges_dict, edges, img3):
    # search parameters
    actions = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
    margin = 1
    
    nodes_to_add = []
    for item in edges.values():
        # print('item = ', item)
        point = item[0]      # item[1] is the coordinates in the dictionary format like [947,598] in nodes_dict = {'0': [947,598]}
        goal = item[1]
        frontier = [[0,[],point]]
        explored = []
        
        found = False
        entry_number = 1

        # for calculating point along an arch
        while not found and len(frontier)>0:
            frontier.reverse()
            candidate = frontier.pop()
            frontier.reverse()
            previous_loc = candidate[1]
            candidate_loc = candidate[2]
            explored.append([previous_loc,candidate_loc])
            if candidate_loc == goal:
                found = True
            for action in actions:
                next_loc = [candidate_loc[0]+action[0],candidate_loc[1]+action[1]]
                if img_base.shape[0] - margin > next_loc[1] > 0 + margin and img_base.shape[1] - margin > next_loc[0] > 0 + margin:

                    # if check for -> checking cell value is black
                    if img_base[next_loc[1]][next_loc[0]] != 255:
                        frontier_loc = [item[2] for item in frontier]
                        explored_loc = [item[1] for item in explored]
                        if next_loc not in explored_loc and next_loc not in frontier_loc:
                            frontier.append([entry_number,candidate_loc,next_loc])
                            entry_number =+ 1
    
        if found:
            # print('found')
            path = [explored[-1][1]]
            previous = explored[-1][0]
            explored_loc = [item[1] for item in explored]
            while previous:
                path.append(previous)
                previous_index = explored_loc.index(previous)
                previous = explored[previous_index][0]
            path.reverse()
            if point not in path:
                path.append(point)
            if goal not in path:
                path.append(goal)
        else:
            print('not found')

        """
        For each edge, the algorithm calculates the length of the edge and the length of the 
        crack pattern. The algorithm checks to see if the length of the crack pattern is more 
        than the length of the edge based on a threshold. The perpendicular distance between the 
        edge and the crack pattern is calculated pixel by pixel. The pixel with the greater distance 
        is selected to add a corner there, and reconnect the edges to refine the graph representation.
        """

        arch = 0
        for i in range(len(path)):
            if i == (len(path)-1):
                break
            arch = arch + np.sqrt(np.square(path[i][0] - path[i+1][0]) + np.square(path[i][1] - path[i+1][1]))
            
        straight = distance(point, goal)
        
        perpendicular_dist = []
        dif_goal_point = [point[0]-goal[0], point[1]-goal[1]]
        for item_pix in path:
            dif_point_pix = [point[0]-item_pix[0], point[1]-item_pix[1]]
            perpendicular_dist.append(np.linalg.norm(np.cross(dif_goal_point, dif_point_pix))/np.linalg.norm(dif_goal_point))
        
        ### DEBUG
        # print(f"IMG_BASE: {img_base.shape}")
        # print(f"EDGE_LEN: {straight}, ARCH_LEN: {0.62 * arch}")
        # print(f"PERP_DIST: {max(perpendicular_dist)}, 0.02MIN_DIS: {min(0.02*img_base.shape[0], 0.02*img_base.shape[1])}")
        # print(f"PERP_DIST: {max(perpendicular_dist)}, 0.05MIN_DIS: {min(0.02*img_base.shape[0], 0.02*img_base.shape[1])}")

        if (straight <= 0.62 * arch and straight >= min(0.005*img_base.shape[0], 0.005*img_base.shape[1]))\
            or (max(perpendicular_dist) >= min(0.002*img_base.shape[0], 0.002*img_base.shape[1])):

            # print("NODES ADDED FOR ARCH MATCHING")
            
            nodes_to_add.append(path[perpendicular_dist.index(max(perpendicular_dist))])
            
    
    nodes.extend(nodes_to_add)
    nodes_dict = list2dict(nodes)
    edges_dict, edges, nodes_dict, nodes = corner_connection(nodes_dict, img_base, nodes)

    return edges_dict, edges, nodes_dict, nodes

# MARK: Corner Connection

def corner_connection(nodes_dict, img_base, nodes):
    # Search parameters
    actions = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
    margin = 1
    
    edges = dict()
    edges_dict = dict()
    e = 0
    for item in nodes_dict.items():
        point = item[1]      # item[1] is the coordinates in the dictionary format like [947,598] in nodes_dict = {'0': [947,598]}
        frontier = [point]
        # print(f"FRONTIER BEGIN: {frontier}")

        explored = []
        while len(frontier) > 0:
            frontier.reverse()
            candidate = frontier.pop()
            frontier.reverse()
            explored.append(candidate)
            
            for action in actions:
                new_pos = [candidate[0]+action[0], candidate[1]+action[1]]
                if img_base.shape[0] - margin > new_pos[1] > 0 + margin and img_base.shape[1] - margin > new_pos[0] > 0 + margin:
                    if img_base[new_pos[1]][new_pos[0]] != 255:
                        if new_pos not in frontier and new_pos not in explored:
                            frontier.append(new_pos)
    
            if candidate in nodes and candidate != point:
                linked_index = getIndex(nodes_dict, candidate)
                if isRepeated(candidate, point, edges) == False:
                    edges[str(e)] = (point, candidate)
                    edges_dict[str(e)] = [point, candidate, distance(point, candidate)]
                    e += 1
    
                ignoreDirection(point, candidate, frontier, window=50)
                
            # print(f"FRONTIER END: {frontier}")
    
    nodes_dict = list2dict(nodes)
    
    edges_dict = loc2idx(edges_dict, nodes_dict)
    return edges_dict, edges, nodes_dict, nodes
    
# MARK: Corner Detection

def corner_detection(thinned, ker_size):
    #################################### Corner Detection Using Shi-Tomasi
    nodes = cv2.goodFeaturesToTrack(thinned, 5000, 0.1, None)
    
    ###### Set the needed parameters to find the refined corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
    ###### Calculate the refined corner locations
    nodes = cv2.cornerSubPix(thinned, nodes, (ker_size, ker_size), (-1,-1), criteria)  # Corners will be centered within a 5by5 window
    
    nodes = np.int_(nodes)
    nodes = [xy for N_i in nodes for xy in N_i]
    
    nodes = np.asarray(nodes)
    nodes = nodes.tolist()
    
    ######## Removing duplicate nodes
    buf_node = []
    for elem in nodes:
        if elem not in buf_node:
            buf_node.append(elem)
    nodes = buf_node
    
    ########################### 
    """
    Are the detected corners actually within the black pixels or might be on the edge 
    within the white pixels?
    """
    ########################### 
    bl_pix_fin = np.where((thinned[:,:] == 0))
    
    temp_l = []
    for i in range(len(bl_pix_fin[0])):
        temp_l.append([bl_pix_fin[1][i], bl_pix_fin[0][i]])
        
    for i in range(len(nodes)):
        if (nodes[i] not in temp_l):
            temp_l = sortByDistance(temp_l, nodes[i])
            nodes[i] = temp_l[0]

    ############ Removing extra nodes in a ker_size by ker_size window
    half = int((ker_size - 1 ) / 2)
    b = [*range(-half, half+1, 1)]
    center = int(((ker_size * ker_size) - 1) /2)
    for node_item in nodes:
        window = []
        for i in b:
            for j in b:
                window.append([node_item[0]+i, node_item[1]+j])
        
        window.remove(window[center])   # removing the center of the created window which is the node itself
        for window_item in window:
            if window_item in nodes:
                nodes.remove(window_item)
    
    nodes_dict = list2dict(nodes)
    
    return nodes, nodes_dict

# MARK: Primary Crack to Graph Algorithm

def crack_to_graph(img_name):
    # SAVE = True
    
    #################################### Loading The Data Using Matplotlib
    img = cv2.imread(img_name)

    #################################### RGB2Gray Using Matplotlib
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # plt.imshow(img_gray, cmap = 'gray')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

    #################################### Defining kernel and padding size
    ker_size_spal = int(np.floor(min(0.01*img_gray.shape[0], 0.01*img_gray.shape[1])))
    if ker_size_spal % 2 == 0:
        ker_size_spal += 1
        
    ker_size_spal = max(ker_size_spal, 5)
    pad_size = 3*ker_size_spal + 1
    
    #################################### Binarization
    retval, img_bin = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY_INV)
    # plt.imshow(img_bin, cmap = 'gray')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

    #################################### Image Padding
    if img_bin.shape[0] % 2 == 1 and img_bin.shape[1] % 2 == 1:
        img_bin = cv2.copyMakeBorder(img_bin, pad_size, pad_size+1, pad_size+1, pad_size, cv2.BORDER_CONSTANT, value=255)
    elif img_bin.shape[0] % 2 == 0 and img_bin.shape[1] % 2 == 1:
        img_bin = cv2.copyMakeBorder(img_bin, pad_size, pad_size, pad_size+1, pad_size, cv2.BORDER_CONSTANT, value=255)
    elif img_bin.shape[0] % 2 == 1 and img_bin.shape[1] % 2 == 0:
        img_bin = cv2.copyMakeBorder(img_bin, pad_size, pad_size+1, pad_size, pad_size, cv2.BORDER_CONSTANT, value=255)
    else :
        img_bin = cv2.copyMakeBorder(img_bin, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=255)

    
    #################################### Morphological Operations / Removing The Spalling Aea
    kernel_spal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ker_size_spal,ker_size_spal))

    img_spal = cv2.dilate(img_bin, kernel = kernel_spal, iterations=2)
    img_spal_subtract = cv2.erode(img_spal, kernel = kernel_spal, iterations=3)
    img_spal = cv2.erode(img_spal, kernel = kernel_spal, iterations=2)
    spalless = cv2.subtract(img_spal_subtract, img_bin)

    # plt.imshow(spalless, cmap = 'gray')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    
    #################################### Image Thinning To Have One-Pixel-Width Crack Pattern
    thinned_spalless = thin(spalless)    # Thinning should be done on the black background image, and white cracks
    thinned_spalless = thinned_spalless.astype(np.uint8)
    
    ##### Where are black and white pixels
    bl_pix_spal = np.where((thinned_spalless[:,:] == 0))
    wh_pix_spal = np.where((thinned_spalless[:,:] == 1))

    ##### Inverting white and black pixels
    thinned_spalless[bl_pix_spal] = [255]
    thinned_spalless[wh_pix_spal] = [0]
    
    ################################### Image Pyramids
    ##### Down sampling the image
    ##### Generate Gaussian pyramid for the gray-scale image
    G = thinned_spalless.copy()
    gp_down = [G]
    for i in range(4):
        G = cv2.pyrDown(G)       # Should be performed on white background image with black crack pixels
        gp_down.append(G)
        
    ##### Up sampling the  / this makes the crack patterns smooth / the jagedness of the cracks is faded
    gp_up = []
    for i in range(4,0,-1):    # range(start, stop, step)
        GE = cv2.pyrUp(gp_down[i])
        gp_up.append(GE)
        # plt.imshow(GE, cmap = 'gray')
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()

    ##### Thinning the up-sampled image To Have One-Pixel-Width Crack Pattern
    gp_up_thinned = []
    for pyr in gp_up:
        retval, pyr = cv2.threshold(pyr, 250, 255, cv2.THRESH_BINARY)
        
        bl_pix_pyr = np.where((pyr[:,:] == 0))
        wh_pix_pyr = np.where((pyr[:,:] == 255))
        pyr[bl_pix_pyr] = [255]
        pyr[wh_pix_pyr] = [0]

        pyr = thin(pyr)
        pyr = pyr.astype(np.uint8)

        bl_pix_pyrfinal = np.where((pyr[:,:] == 0))
        wh_pix_pyrfinal = np.where((pyr[:,:] == 1))
        pyr[bl_pix_pyrfinal] = [255]
        pyr[wh_pix_pyrfinal] = [0]
        
        gp_up_thinned.append(pyr)
        

    ################################### Morphological Operations
    ker_size_morph = int(np.floor(min(0.004*img_gray.shape[0], 0.004*img_gray.shape[1])))
    if ker_size_morph % 2 == 0:
        ker_size_morph += 1

    ker_size_morph = max(ker_size_morph, 5)
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ker_size_morph,ker_size_morph))
    
    crack_pix_init_orig = len(wh_pix_spal[0])
    crack_pix_init_smooth = len(wh_pix_pyrfinal[0])
    # change_orig_smooth = ((abs(crack_pix_init_orig - crack_pix_init_smooth)/crack_pix_init_orig) * 100)
    
    change = -1
    i = 1

    start = time.time()

    while change < 15: # limiting the crack pattern smoothing to 15 %
        ################################ Erosion and Dilation
        ##### Binarizing the image
        retval, img_bin = cv2.threshold(gp_up_thinned[3], 250, 255, cv2.THRESH_BINARY)
        
        ##### Eroding the image
        img_erosion = cv2.erode(img_bin, kernel = kernel_morph, iterations = i)
        
        ##### Dilating the image
        img_dilation = cv2.dilate(img_erosion, kernel = kernel_morph, iterations = i-2)
        
        ##### Inverting the black and white pixels for the thinning process
        bl_pix_dil = np.where((img_dilation[:,:] == 0))
        wh_pix_dil = np.where((img_dilation[:,:] == 255))
        img_dilation[bl_pix_dil] = [255]
        img_dilation[wh_pix_dil] = [0]        
    
        thinned = thin(img_dilation)
        thinned = thinned.astype(np.uint8)
    
        bl_pix_final = np.where((thinned[:,:] == 0))
        wh_pix_final = np.where((thinned[:,:] == 1))
        thinned[bl_pix_final] = [255]
        thinned[wh_pix_final] = [0]
        
        crack_pix_fin = len(wh_pix_final[0])
        # change_orig = ((abs(crack_pix_init_orig - crack_pix_fin)/crack_pix_init_orig) * 100)
        change_smooth = ((abs(crack_pix_init_smooth - crack_pix_fin)/crack_pix_init_smooth) * 100)
        
        # print("Change = %", round(change,1))
        # print("i = ", i)
        change = change_smooth
        
        if i == 5: # limiting the crack pattern smoothing to 5 cycles
            break
        
        i = i + 1

    end = time.time()
    
    # print(f"crack smoothing time: {end-start} s")
        
    bl_pix_fin = np.where((thinned[:,:] == 0))
    
    ##### percentage of change in the number of crack pixels due to morphological operations
    change = ((abs(crack_pix_init_orig - len(bl_pix_fin[0]))/crack_pix_init_orig) * 100)
    # print(f"Crack Pixel Change = {round(change,1)}%")
    
    #################################### Deep Copy For Final Plot
    img1 = deepcopy(thinned)
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
    img2 = deepcopy(thinned)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)

    #################################### Corner Detection Using Shi-Tomasi
    nodes, nodes_dict = corner_detection(thinned, ker_size_morph)
    # print("Corners are detected using the Shi-Tomasi algorithm!")
    
    #################################### Connecting Corners As Edges
    ang_bet_lines = 155

    # start = time.time()

    while True:
        #################################### Connecting the Detected Corners As Edges
        edges_dict, edges, nodes_dict, nodes = corner_connection(nodes_dict, thinned, nodes)
        len_nodes_init = len(nodes)

        ## TODO: UNDERSTAND THIS ALGORITHM & EXTRACT THE IN-BETWEEN ANGLE 
        #################################### Refining connected corners with only two edges roughly straight
        ## If two edges are connected together and their in-between angle is close to 180 degrees, it means these edges can roughly be one
        ## straight edge. Therefore, the node between these edges will be removed.
        ## This is done in a while loop so to make sure no two straight lines are left.
        nodes, nodes_dict = node_remv_straight_line(nodes, edges_dict, edges, ang_bet_lines)
        ang_bet_lines = min(ang_bet_lines, 175)
        ang_bet_lines = ang_bet_lines + 10
        ##### convert the nodes to a dictionary
        nodes_dict = list2dict(nodes)
    
        len_nodes_new = len(nodes)
        
        if len_nodes_new == len_nodes_init:
            break

    # end = time.time()

    # print(f"straight line node detection time: {end-start} s")

    # print("The initial graph representation is created!")
    
    ####################### Removing Single Nodes
    nodes, nodes_dict = single_node_remover(nodes, edges_dict)
    edges_dict, edges, nodes_dict, nodes = corner_connection(nodes_dict, thinned, nodes)
    # print("Single nodes are removed!")
    
    #################### Removing single degree nodes with length shorter than a threshhold
    # nodes, nodes_dict = short_spal_edge_removal(nodes, edges_dict, edges, thinned, img_spal, kernel_spal)
    # edges_dict, edges, nodes_dict, nodes = Corner_Connection(nodes_dict, thinned, nodes)
    # print("i=4")
    
    ####################### Removing Single Nodes
    # nodes, nodes_dict = single_node_remover (nodes, edges_dict)
    # edges_dict, edges, nodes_dict, nodes = Corner_Connection(nodes_dict, thinned, nodes)
    # print("i=5")
    
    ####################### Arch Matching
    ## There are some parts of crack patterns that forms an arch, and the corner detection algorithm cannot detect a corner when the turn is not sharp.
    ## This part tries address this issue to match the arch as much as possible.

    start = time.time()

    i = 0
    while True:
        len_nodes_init = len(nodes)
        edges_dict, edges, nodes_dict, nodes = arch_match(nodes_dict, thinned, nodes, edges_dict, edges, img2)
        i = i + 1
        len_nodes_new = len(nodes)
        
        if len_nodes_new == len_nodes_init or i == 5:
            break

    end = time.time()

    # print(f"arch matching time: {end-start} s")
    # print("The archs are matched!")
    # print("The final graph representation of the crack pattern is ready!")

    return edges, edges_dict, nodes, nodes_dict, img_bin, img1

# MARK: Draw/Plot of Graph Result

def draw_graph(img_bin, img_plot, edges, nodes, img_name, img_format='png', save_plot=False):
    ##### draw circles on all nodes
    circle_thickness = max(int(np.floor(min(0.004*img_bin.shape[0], 0.004*img_bin.shape[1]))),3)
    for i in nodes:
        x, y = i[0], i[1]
        cv2.circle(img_plot, (x, y), circle_thickness, (0,255,0), -1)    # draw circles on the corner

    # ##### Resulting Image
    for value in edges.values():
        cv2.line(img_plot, value[0], value[1], (255,0,0), thickness=1, lineType=8)

    # fig, ax = plt.subplots()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_plot, cmap = None)
    image_format = img_format # e.g .png, .svg, etc.
    image_name = img_name+"_connected."+image_format

    if save_plot:
        save_folder_path = "./CracktoGraph/"
        save_path = save_folder_path + image_name
        fig.savefig(save_path, format=image_format, dpi=1200)

# MARK: Graph Feature Extraction

def feature_extraction(edges_dict, nodes_dict):
    PLOT = False
    
    # read the "nodes" in the crack graph
    nodes = dict()
    k = 0
    for node in nodes_dict.items():
       k, v = node
       nodes[k] = v

    total_nodes = int(k) + 1     # total number of nodes in the crack graph

    
    # read the "edges" dictionary exported by main_1.py
    edges = dict()
    for edge in edges_dict.items():
       k, v = edge
       edges[k] = v
    total_edges = int(k) + 1    # total number of nodes in the crack graph

    
    #################### Making the connectivity (adjacency) matrix
    adj_matrix = adj_mat(nodes, edges)
 
    ########################### Feature 1: node degree #####################################################################
    k, k_w = nodeDegree(adj_matrix)
    k_avg = np.mean(k)
    kw_avg = np.mean(k_w)


    # ########################### Feature 2: shortest path ###################################################################
    # d_w = np.zeros((total_nodes, total_nodes), dtype=float)
    # for n_th in range(total_nodes):
    #     d_w[n_th] = dijkstra(total_nodes, adj_matrix, n_th)     # update row n-th by shortest path from Dijkstra algorithm

    ########################### Feature 3: number of triangles #############################################################
    # t, t_w = triangles(adj_matrix)
        # t_avg = np.mean(t)
    # tw_avg = np.mean(t_w)


    ########################### Feature 4: Network's clustering coefficient ################################################
    # C = clusteringCoeff(k, t)
    # C_w = clusteringCoeff(k, t_w)


    ########################### Feature 5: Network's transitivity ##########################################################
    # T = transitivity(k, t)
    # T_w = transitivity(k, t_w)


    ########################### Feature 6: Network's global efficiency #####################################################
    ############ making the graph
    G = nx.Graph(adj_matrix, nodetype=int)
    E = nx.global_efficiency(G)
    # E = globalEff(d_w)


    ########################### Feature 7: Network's local efficiency ######################################################
    E_loc = nx.local_efficiency(G)
    # E_loc = localEff(adj_matrix, k, d_w)

    
    ########################### Feature 8: Network's Max and Min Eigenvalue ######################################################
    # e = np.linalg.eigvals(adj_matrix)
    # e_real = [ele.real for ele in e]
    # e_imag = [ele.imag for ele in e]
    # deg_cen = nx.degree_centrality(G)
    

    return k_avg, kw_avg, E, E_loc

# MARK: Removing node on Straing Line Edges
   
def node_remv_straight_line(nodes, edges_dict, edges, ang_bet_lines):

    adj_matrix = adj_mat(nodes, edges_dict)
    
    k, k_w = nodeDegree(adj_matrix)

    twoDegree_cord = []
    for idx_node, item_node in enumerate(nodes):
        if k[idx_node] == 2:
            x1 = item_node[0]
            y1 = item_node[1]
            
            V_lst = []
            for edges_item in edges.values():
                if item_node == edges_item[0] or item_node == edges_item[1]:
                
                    if edges_item[1] != item_node:
                        x2 = edges_item[1][0]
                        y2 = edges_item[1][1]
                    else:
                        x2 = edges_item[0][0]
                        y2 = edges_item[0][1]
                    
                    V_lst.append([(x1 - x2), (y1 - y2)])
                    
            def dot(vA, vB):
                return vA[0]*vB[0]+vA[1]*vB[1]
            
            # Get vector form
            vA = V_lst[0]
            vB = V_lst[1]
            # Get dot prod
            dot_prod = dot(vA, vB)
            # Get magnitudes
            magA = dot(vA, vA)**0.5
            magB = dot(vB, vB)**0.5
            # Get cosine value
            cos_ = dot_prod/magA/magB
            # Get angle in radians and then convert to degrees
            angle = math.acos(round(cos_,3))
            # Basically doing angle <- angle mod 360
            ang_deg = math.degrees(angle)%360
        
            if ang_deg - 180 >= 0:
                ang_deg = 360 - ang_deg
                        
            if ang_deg > ang_bet_lines:
                twoDegree_cord.append(item_node)
    
    
    for item in twoDegree_cord:
        nodes.remove(item)
        
    nodes_dict = list2dict(nodes)    

    return nodes, nodes_dict

# MARK: Short spal from edge removal

def short_spal_edge_removal(nodes, edges_dict, edges, thinned, img_spal, kernel_spal):

    spal_pix_loc = np.where((img_spal[:,:] == 0))
    spal_pix = []
    for i in range(len(spal_pix_loc[0])):
        spal_pix.append([spal_pix_loc[1][i], spal_pix_loc[0][i]])
    
    
    node_to_remove = []
    for idx_node, item_node in enumerate(nodes):
        if item_node in spal_pix and k[idx_node] == 1:
    
            for edges_idx, edges_item in edges.items():
                if item_node == edges_item[0] or item_node == edges_item[1]:
                    if edges_item[1] != item_node:
                        x2y2 = edges_item[1]
                    else:
                        x2y2 = edges_item[0]
                    if distance(item_node, x2y2) < min(0.03*thinned.shape[0], 0.03*thinned.shape[1]) and item_node not in node_to_remove:
                    # if item_node not in node_to_remove:
                        node_to_remove.append(item_node)

    ######## Removing nodes with degree = 1
    for item in node_to_remove:
        nodes.remove(item)

    nodes_dict = list2dict(nodes)    

    return nodes, nodes_dict

# MARK: Single node remover

def single_node_remover(nodes, edges_dict):

    adj_matrix = adj_mat(nodes, edges_dict)
    
    k, k_w = nodeDegree(adj_matrix)

    node_to_remove = []
    for idx_node, item_node in enumerate(nodes):
        if k[idx_node] == 0:
            node_to_remove.append(item_node)

    for item in node_to_remove:
        nodes.remove(item)
        
    nodes_dict = list2dict(nodes)
    
    return nodes, nodes_dict