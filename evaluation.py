# encoding: utf-8

import os
import shutil
from copy import deepcopy
import argparse
import setproctitle
import scipy.stats
import numpy as np
from collections import Counter, defaultdict
from math import radians, cos, sin, asin, sqrt
import pandas as pd
from tqdm import tqdm
from utils import set_seed, divide_grids_by_num, divide_grids_by_size, read_data_from_file
import networkx as nx
import time
import json
# from openlocationcode import openlocationcode as olc


def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) 
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 
    # distance=round(distance/1000,3)
    return distance                                             # m


class EvalUtils(object):
    """
    some commonly-used evaluation tools and functions
    """

    @staticmethod
    def filter_zero(arr):
        """
        remove zero values from an array
        :param arr: np.array, input array
        :return: np.array, output array
        """
        arr = np.array(arr)
        filtered_arr = np.array(list(filter(lambda x: x != 0., arr)))
        return filtered_arr

    @staticmethod
    def arr_to_distribution(arr, min, max, bins):
        """
        convert an array to a probability distribution
        :param arr: np.array, input array
        :param min: float, minimum of converted value
        :param max: float, maximum of converted value
        :param bins: int, number of bins between min and max
        :return: np.array, output distribution array
        """
        # distribution, base = np.histogram(
        #     arr, np.arange(
        #         min, max, float(
        #             max - min) / bins))
        distribution, base = np.histogram(
            arr, np.linspace(
                min, max, bins))
        return distribution, base[:-1]

    @staticmethod
    def norm_arr_to_distribution(arr, bins=100):
        """
        normalize an array and convert it to distribution
        :param arr: np.array, input array
        :param bins: int, number of bins in [0, 1]
        :return: np.array, np.array
        """
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = EvalUtils.filter_zero(arr)
        # distribution, base = np.histogram(arr, np.arange(0, 1, 1. / bins))
        distribution, base = np.histogram(arr, np.linspace(0, 1, bins))
        return distribution, base[:-1]

    @staticmethod
    def log_arr_to_distribution(arr, min=-30., bins=100):
        """
        calculate the logarithmic value of an array and convert it to a distribution
        :param arr: np.array, input array
        :param bins: int, number of bins between min and max
        :return: np.array,
        """
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = EvalUtils.filter_zero(arr)
        arr = np.log(arr)
        distribution, base = np.histogram(arr, np.linspace(min, 0., bins))
        return distribution, base[:-1]
    
        ret_dist, ret_base = [], []
        for i in range(bins-1):
            if int(distribution[i]) == 0:
                continue
            else:
                ret_dist.append(distribution[i])
                ret_base.append(base[i])
        return np.array(ret_dist), np.array(ret_base)

    @staticmethod
    def get_js_divergence(p1, p2):
        """
        calculate the Jensen-Shanon Divergence of two probability distributions
        :param p1:
        :param p2:
        :return:
        """
        # normalize
        p1 = p1 / (p1.sum()+1e-14)
        p2 = p2 / (p2.sum()+1e-14)
        m = (p1 + p2) / 2
        js = 0.5 * scipy.stats.entropy(p1, m) + \
            0.5 * scipy.stats.entropy(p2, m)
        return js
    


class IndividualEval(object):

    def __init__(self, path, max_len=20, boundary=None, grid_num=64, grid_size=0.1, filename=None, weighted=False):
        self.nodes = pd.read_csv(os.path.join(path, 'nodes.csv'), header=0, usecols=['osmid', 'y', 'x', 'fid'])
        self.edges = pd.read_csv(os.path.join(path, 'edges.csv'), header=0, usecols=['u', 'v', 'length', 'fid', 'geometry'])
        self.path = path
        self.node_num, self.edge_num = len(self.nodes), len(self.edges)
        self.max_loc = self.edge_num + 1
        self.seq_len = max_len
        self.boundary = boundary
        self.grid_num = grid_num
        self.grid_size = grid_size 
        self.filename = filename
        
        self.node2id = {}
        self.node2gps = {}
        
        self.edge2len = defaultdict(float)
        self.edge2id = {}
        self.edge2gps = {}
        self.edge2code = {}
        self.edge2gpslist = defaultdict(list)
        
        self.road_network = np.zeros((self.node_num, self.node_num), dtype=np.float32)
        self.segment_graph = np.zeros((self.max_loc, self.max_loc), dtype=np.float32) # for start token 0
        self.weighted = weighted
        
        # self.neighbors = defaultdict(set)
        
        self.read_nodes()
        self.read_edges()
        self.construct_road_network()
        self.construct_segment_graph()
        self.construct_weighted_segment_graph()
        
        # self.max_distance = self.get_max_distance()                                                 # consume much time
        if 'porto' in self.path:
            self.max_distance = 10000                                         
        elif 'cd' in self.path:
            self.max_distance = 20000  
        elif 'sz' in self.path:
            self.max_distance = 20000  
        else:
            self.max_distance = 20000  
        
    def read_nodes(self):
        for node in self.nodes.itertuples():
            osm_id = node.osmid                 # int
            node_id = node.fid
            lng, lat = node.x, node.y            
            self.node2id[osm_id] = node_id
            self.node2gps[node_id] = (lng, lat)
            
    
    def read_edges(self):
        for edge in self.edges.itertuples():
            source, target = edge.u, edge.v                      # int
            edge_len, edge_id = edge.length, edge.fid + 1        # edge id starts from 1
            source, target = self.node2id[source], self.node2id[target]
            if (source, target) not in self.edge2id.keys():
                self.edge2id[(source, target)] = edge_id            
            else:
                # print(source, target, self.edge2id[(source, target)], edge_id)
                # print(edge)
                pass
            self.edge2len[edge_id] = edge_len
            edge_geometry = edge.geometry[12:-1].split(',')                     # e.g., 'LINESTRING (-8.6406364 41.1660713, -8.6409114 41.1662762)' -> ['-8.6406364 41.1660713', ' -8.6409114 41.1662762']
            gps_list = [coord.strip().split(' ') for coord in edge_geometry]            # list of list, e.g., [[lng1, lat1], [lng2, lat2], ...]
            self.edge2gpslist[edge_id] += gps_list
            if len(gps_list) == 2:
                self.edge2gps[edge_id] = ((float(gps_list[0][0])+float(gps_list[1][0]))/2, (float(gps_list[0][1])+float(gps_list[1][1]))/2)
            elif len(gps_list) % 2 == 0:
                lng_left, lat_left = gps_list[len(edge_geometry)//2-1]
                lng_right, lat_right = gps_list[len(edge_geometry)//2]
                lng, lat = (float(lng_left) + float(lng_right)) / 2, (float(lat_left) + float(lat_right)) / 2
                self.edge2gps[edge_id] = (lng, lat)
            else:
                lng, lat = gps_list[len(edge_geometry)//2]
                self.edge2gps[edge_id] = (float(lng), float(lat))      
    
    def construct_road_network(self):
        for edge in self.edges.itertuples():
            source, target = edge.u, edge.v                      # string
            source, target = self.node2id[source], self.node2id[target]
            if (source, target) in self.edge2id.keys():
                self.road_network[source][target] = 1
    
    def construct_segment_graph(self):
        G = nx.Graph()
        for edge in self.edges.itertuples():
            source, target = edge.u, edge.v    
            edge_id = edge.fid + 1
            neigh_edges = self.edges.loc[self.edges.u == target].fid + 1
            for n_edge in neigh_edges:
                self.segment_graph[edge_id][n_edge] = 1

        # self.segment_graph[0, 1:] = 1
        for i in range(1, self.segment_graph.shape[0]):
            if self.segment_graph[i].sum():
                self.segment_graph[0, i] = 1                    
        
        # store for node2vec
        unweighted_filename = self.filename
        if not os.path.exists(unweighted_filename):
            for i in range(0, self.segment_graph.shape[0]):
                for j in range(0, self.segment_graph.shape[0]):
                    if self.segment_graph[i][j]:
                        G.add_edge(i, j)
            for k in range(0, self.segment_graph.shape[0]):  # isolated nodes
                if self.segment_graph[k].sum() == 0 and self.segment_graph[:, k].sum() == 0:
                    G.add_edge(k, k)
            nx.write_edgelist(G, self.filename, data=False)
    
    def construct_weighted_segment_graph(self):
        file_dir, file_name = os.path.split(self.filename)
        prefix, suffix = file_name.split('.')
        weighted_filename = os.path.join(file_dir, prefix + '_weighted.' + suffix)
        weighted_segment_graph = deepcopy(self.segment_graph)
        train_trajs = np.load(self.path+'/train.npy', allow_pickle=True).astype(np.int32)  
        for traj in train_trajs:
            for i in range(self.seq_len - 1):
                weighted_segment_graph[traj[i]+1][traj[i+1]+1] += 1   
        self.weighted_segment_graph = weighted_segment_graph
        
        if not os.path.exists(weighted_filename):
            G = nx.Graph()   
            for i in range(0, weighted_segment_graph.shape[0]):
                for j in range(0, weighted_segment_graph.shape[0]):
                    if weighted_segment_graph[i][j]:
                        G.add_edge(i, j, weight=weighted_segment_graph[i][j])
            for k in range(0, weighted_segment_graph.shape[0]): 
                if weighted_segment_graph[k].sum() == 0 and weighted_segment_graph[:, k].sum() == 0:
                    G.add_edge(k, k, weight=1.0)
            
            nx.write_weighted_edgelist(G, weighted_filename)                                   
                     
    def get_max_distance(self):
        edge_max = max(list(self.edge2len.values()))                            #   unit: meter
        with tqdm(total=self.edge_num) as pbar:
            for edge_i in self.edge2gps.keys():
                for edge_j in self.edge2gps.keys():
                    if edge_i != edge_j:
                        lng_i, lat_i = self.edge2gps[edge_i]
                        lng_j, lat_j = self.edge2gps[edge_j]
                        distance = geodistance(lng_i, lat_i, lng_j, lat_j)          #   unit: meter
                        if distance > edge_max:
                            edge_max = distance
                pbar.update(1)
                        
        return edge_max
        
    def get_topk_visits(self,trajs, k):
        topk_visits_loc = []
        topk_visits_freq = []
        for traj in trajs:
            topk = Counter(traj).most_common(k)
            for i in range(len(topk), k):
                # supplement with (loc=-1, freq=0)
                topk += [(-1, 0)]
            loc = [l for l, _ in topk]
            freq = [f for _, f in topk]
            loc = np.array(loc, dtype=int)
            freq = np.array(freq, dtype=float) / trajs.shape[1]
            topk_visits_loc.append(loc)
            topk_visits_freq.append(freq)
        topk_visits_loc = np.array(topk_visits_loc, dtype=int)
        topk_visits_freq = np.array(topk_visits_freq, dtype=float)
        return topk_visits_loc, topk_visits_freq

    
    def get_overall_topk_visits_freq(self, trajs, k):
        _, topk_visits_freq = self.get_topk_visits(trajs, k)
        mn = np.mean(topk_visits_freq, axis=0)
        return mn / np.sum(mn)


    def get_overall_topk_visits_loc_freq_arr(self, trajs, k=1):
        topk_visits_loc, _ = self.get_topk_visits(trajs, k)
        k_top = np.zeros(self.max_loc, dtype=float)
        for i in range(k):
            cur_k_visits = topk_visits_loc[:, i]
            for ckv in cur_k_visits:
                index = int(ckv)
                if index == -1:
                    continue
                k_top[index] += 1
        k_top = k_top / np.sum(k_top)
        return k_top

    
    def get_overall_topk_visits_loc_freq_dict(self, trajs, k):
        topk_visits_loc, _ = self.get_topk_visits(trajs, k)
        k_top = {}
        for i in range(k):
            cur_k_visits = topk_visits_loc[:, i]
            for ckv in cur_k_visits:
                index = int(ckv)
                if index in k_top:
                    k_top[int(ckv)] += 1
                else:
                    k_top[int(ckv)] = 1
        return k_top

    def get_overall_topk_visits_loc_freq_sorted(self, trajs, k):
        k_top = self.get_overall_topk_visits_loc_freq_dict(trajs, k)
        k_top_list = list(k_top.items())
        k_top_list.sort(reverse=True, key=lambda k: k[1])
        return np.array(k_top_list)


    def get_geodistances(self, trajs):
        distances = []
        seq_len = 48
        for traj in trajs:
            for i in range(seq_len - 1):
                lng1 = self.X[traj[i]]
                lat1 = self.Y[traj[i]]
                lng2 = self.X[traj[i + 1]]
                lat2 = self.Y[traj[i + 1]]
                distances.append(geodistance(lng1,lat1,lng2,lat2))
        distances = np.array(distances, dtype=float)
        return distances
    
    def get_distances(self, trajs):
        distances = []
        for traj in trajs:
            for i in range(self.seq_len):
                distances.append(self.edge2len[traj[i]])
                
        distances = np.array(distances, dtype=float)
        return distances
    
    def get_durations(self, trajs):
        d = []
        for traj in trajs:
            num = 1
            for i, lc in enumerate(traj[1:]):
                if lc == traj[i]:
                    num += 1
                else:
                    d.append(num)
                    num = 1
        return np.array(d)/self.seq_len
    
    def get_gradius(self, trajs):
        """
        get the std of the distances of all points away from center as `gyration radius`
        :param trajs:
        :return:
        """
        gradius = []
        for traj in trajs:
            xs, ys = [], []
            for t in traj:
                for i, (lng, lat) in enumerate(self.edge2gpslist[t]):
                    lng, lat = float(lng), float(lat)
                    xs.append(lng)
                    ys.append(lat)
            xs, ys = np.array(xs), np.array(ys)
            xcenter, ycenter = np.mean(xs), np.mean(ys)
            dxs = xs - xcenter
            dys = ys - ycenter
            rad = [dxs[i]**2 + dys[i]**2 for i in range(len(dxs))]
            rad = np.mean(np.array(rad, dtype=float))
            gradius.append(rad)
        gradius = np.array(gradius, dtype=float)
        return gradius
    
    def get_periodicity(self, trajs):
        """
        stat how many repetitions within a single trajectory
        :param trajs:
        :return:
        """
        reps = []
        for traj in trajs:
            reps.append(float(len(set(traj)))/self.seq_len)
        reps = np.array(reps, dtype=float)
        return reps

    def get_geogradius(self, trajs):
        """
        get the std of the distances of all points away from center as `gyration radius`
        :param trajs:
        :return:
        """
        gradius = []
        for traj in trajs:
            xs = np.array([self.X[t] for t in traj])
            ys = np.array([self.Y[t] for t in traj])
            lng1, lat1 = np.mean(xs), np.mean(ys)
            rad = []
            for i in range(len(xs)):                   
                lng2 = xs[i]
                lat2 = ys[i]
                distance = geodistance(lng1,lat1,lng2,lat2)
                rad.append(distance)
            rad = np.mean(np.array(rad, dtype=float))
            gradius.append(rad)
        gradius = np.array(gradius, dtype=float)
        return gradius
    
    def get_density(self, trajs, use_size=False):
        if use_size:
            lat_size, lng_size, lat_grid_num, lng_grid_num = divide_grids_by_size(self.boundary, self.grid_size)             # longitude as rows and latitude as columns
            density = np.zeros((lat_grid_num*lng_grid_num), dtype=float)
            with tqdm(total=len(trajs)) as pbar:
                for traj in trajs:
                    for t in traj:
                        prev_grid = -1
                        gps_list = self.edge2gpslist[t]                                              # list of list, e.g., [[lng1, lat1], [lng2, lat2], ...]
                        for i, (lng, lat) in enumerate(gps_list):
                            lng, lat = float(lng), float(lat)
                            # clip
                            lng = max(min(lng, self.boundary['max_lng']), self.boundary['min_lng'])  
                            lat = max(min(lat, self.boundary['max_lat']), self.boundary['min_lat'])
                            grid_i = int((lat - self.boundary['min_lat']) / lat_size)                        # row
                            grid_j = int((lng - self.boundary['min_lng']) / lng_size)                        # col
                            curr_grid = grid_i * lng_grid_num + grid_j
                            if curr_grid != prev_grid:
                                prev_grid = curr_grid
                                density[curr_grid] += 1    
                    pbar.update(1)
        else:
            grids_num, latgrids, longrids = divide_grids_by_num(self.boundary, self.grid_num)  
            density = np.zeros((grids_num*grids_num), dtype=float)
            with tqdm(total=len(trajs)) as pbar:
                for traj in trajs:
                    for t in traj:
                        prev_grid = -1
                        gps_list = self.edge2gpslist[t]                                              # list of list, e.g., [[lng1, lat1], [lng2, lat2], ...]
                        for i, (lng, lat) in enumerate(gps_list):
                            grid_i, grid_j = 0, 0
                            lng, lat = float(lng), float(lat)
                            for j in range(grids_num):
                                if lng < longrids[j]:                                                   # [ ) left open, right close        
                                    grid_i = j - 1
                                    break
                                elif lng == longrids[j]:
                                    grid_i = j
                                    break
                            for k in range(grids_num):
                                if lat < latgrids[k]:                                                   # [ ) left open, right close        
                                    grid_j = k - 1
                                    break
                                elif lat == latgrids[k]:
                                    grid_j = k
                                    break
                            curr_grid = grid_i * grids_num + grid_j
                            if curr_grid != prev_grid:
                                prev_grid = curr_grid
                                density[curr_grid] += 1  
                    pbar.update(1)
            
        return density / np.sum(density)
    
    def get_odflow(self, trajs):
        density = np.zeros((self.max_loc, self.max_loc), dtype=float)
        for traj in trajs:
            start, end = traj[0], traj[-1]
            density[start][end] += 1
        density = density.flatten()
        
        return density / np.sum(density)
        return density
    
    def get_single_odflow(self, trajs):
        density_o = np.zeros((self.max_loc), dtype=float)
        density_d = np.zeros((self.max_loc), dtype=float)
        for traj in trajs:
            start, end = traj[0], traj[-1]
            density_o[start] += 1
            density_d[end] += 1
        density_o = density_o.flatten()
        density_d = density_d.flatten()
        
        return density_o / np.sum(density_o), density_d / np.sum(density_d)
    
    def get_connectivity(self, trajs):
        count = 0
        total = len(trajs)
        count_partial = 0
        total_partial = total * (self.seq_len - 1)
        for traj in trajs:
            flag = True
            for i in range(self.seq_len - 1):
                if not self.segment_graph[traj[i]][traj[i+1]]:                                          # i.e., connectivity
                    flag = False
                else:
                   count_partial += 1 
            if flag:    
                count += 1
                
        return count / total, count_partial / total_partial
    
    def get_gravity(self, trajs, use_size=False):
        if use_size:
            lat_size, lng_size, lat_grid_num, lng_grid_num = divide_grids_by_size(self.boundary, self.grid_size)             # longitude as rows and latitude as columns
            density = np.zeros((lat_grid_num*lng_grid_num), dtype=float)
            gravity_density = np.zeros((lat_grid_num*lng_grid_num, lat_grid_num*lng_grid_num), dtype=float)
            with tqdm(total=len(trajs)) as pbar:
                for traj in trajs:
                    start, end = traj[0], traj[-1]
                    for t in [start, end]:
                        prev_grid = -1
                        gps_list = self.edge2gpslist[t]                                              # list of list, e.g., [[lng1, lat1], [lng2, lat2], ...]
                        for i, (lng, lat) in enumerate(gps_list):
                            lng, lat = float(lng), float(lat)
                            grid_i = int((lat - self.boundary['min_lat']) / lat_size)                        # row
                            grid_j = int((lng - self.boundary['min_lng']) / lng_size)                        # col
                            curr_grid = grid_i * lng_grid_num + grid_j
                            if curr_grid != prev_grid:
                                prev_grid = curr_grid
                                density[curr_grid] += 1   
                    pbar.update(1) 
        else:
            grids_num, latgrids, longrids = divide_grids_by_num(self.boundary, self.grid_num)  
            density = np.zeros((grids_num*grids_num), dtype=float)
            gravity_density = np.zeros((grids_num*grids_num, grids_num*grids_num), dtype=float)
            with tqdm(total=len(trajs)) as pbar:
                for traj in trajs:
                    start, end = traj[0], traj[-1]
                    for t in [start, end]:
                        prev_grid = -1
                        gps_list = self.edge2gpslist[t]                                              # list of list, e.g., [[lng1, lat1], [lng2, lat2], ...]
                        for i, (lng, lat) in enumerate(gps_list):
                            grid_i, grid_j = 0, 0
                            lng, lat = float(lng), float(lat)
                            for j in range(grids_num):
                                if lng < longrids[j]:                                                   # [ ) left open, right close        
                                    grid_i = j - 1
                                    break
                                elif lng == longrids[j]:
                                    grid_i = j
                                    break
                            for k in range(grids_num):
                                if lng < latgrids[k]:                                                   # [ ) left open, right close        
                                    grid_j = k - 1
                                    break
                                elif lng == latgrids[k]:
                                    grid_j = k
                                    break
                            curr_grid = grid_i * grids_num + grid_j
                            if curr_grid != prev_grid:
                                prev_grid = curr_grid
                                density[curr_grid] += 1  
                    pbar.update(1)
        
        def inverse_mapping(index):
            if use_size:
                grid_y = index % lng_grid_num                                                              # j-th column                     
                grid_x = (index - grid_y) // lng_grid_num                                                  # i-th row
                lng = lng_size * grid_y + self.boundary['min_lng']
                lat = lat_size * grid_x + self.boundary['min_lat']
            else:
                grid_y = index % grids_num  
                grid_x = (index - grid_y) // grids_num     
                lng = longrids[grid_y]
                lat = latgrids[grid_x]
                
            return lng, lat
        
        for i in range(density.shape[0]):
            lng_i, lat_i = inverse_mapping(i)
            for j in range(density.shape[0]):
                lng_j, lat_j = inverse_mapping(j)
                gravity_density[i][j] = density[i] * density[j] / (geodistance(lng_i, lat_i, lng_j, lat_j) / 1000 + 1e-6)
        gravity_density = gravity_density.flatten()
           
        return density / np.sum(density)

    def get_individual_jsds(self, t1, t2):
        """
        get jsd scores of individual evaluation metrics
        :param t1: test_data
        :param t2: gene_data
        :return:
        """
        # d1 = self.get_distances(t1)
        # d2 = self.get_distances(t2)
        
        # d1_dist, _ = EvalUtils.arr_to_distribution(
        #     d1, 0, self.max_distance, 1000)
        # d2_dist, _ = EvalUtils.arr_to_distribution(
        #     d2, 0, self.max_distance, 1000)
        # distance_jsd = EvalUtils.get_js_divergence(d1_dist, d2_dist)
        distance_jsd = 0.
        

        g1 = self.get_gradius(t1)
        g2 = self.get_gradius(t2)
        g1_dist, _ = EvalUtils.arr_to_distribution(
            g1, 0, 1, 10000)
        g2_dist, _ = EvalUtils.arr_to_distribution(
            g2, 0, 1, 10000)
        radius_jsd = EvalUtils.get_js_divergence(g1_dist, g2_dist)
        
        # g1_dist, _ = EvalUtils.log_arr_to_distribution(
        #     g1, -10, 1000)
        # g2_dist, _ = EvalUtils.log_arr_to_distribution(
        #     g2, -10, 1000)
        # radius_jsd = EvalUtils.get_js_divergence(g1_dist, g2_dist)
        
        
        
        # du1 = self.get_durations(t1)                                                
        # du2 = self.get_durations(t2)     
        # du1_dist, _ = EvalUtils.arr_to_distribution(du1, 0, 1, self.seq_len)
        # du2_dist, _ = EvalUtils.arr_to_distribution(du2, 0, 1, self.seq_len)
        # duration_jsd = EvalUtils.get_js_divergence(du1_dist, du2_dist)
        
        # p1 = self.get_periodicity(t1)
        # p2 = self.get_periodicity(t2)
        # p1_dist, _ = EvalUtils.arr_to_distribution(p1, 0, 1, self.seq_len+1)
        # p2_dist, _ = EvalUtils.arr_to_distribution(p2, 0, 1, self.seq_len+1)
        # dailyloc_jsd = EvalUtils.get_js_divergence(p1_dist, p2_dist)

        
        l1 =  CollectiveEval.get_visits(t1, self.max_loc)
        l2 =  CollectiveEval.get_visits(t2, self.max_loc)
        loc1_dist, _ = EvalUtils.arr_to_distribution(
            l1, 0, 1, 10000)
        loc2_dist, _ = EvalUtils.arr_to_distribution(
            l2, 0, 1, 10000) 
        location_jsd = EvalUtils.get_js_divergence(loc1_dist, loc2_dist) 
        
        l1_dist, _ = CollectiveEval.get_topk_visits(l1, 100)
        l2_dist, _ = CollectiveEval.get_topk_visits(l2, 100)
        l1_gdist, _ = EvalUtils.arr_to_distribution(l1_dist, 0, 1, 5000)
        l2_gdist, _ = EvalUtils.arr_to_distribution(l2_dist, 0, 1, 5000) 
        g_rank_jsd = EvalUtils.get_js_divergence(l1_gdist, l2_gdist)  

        f1 = self.get_overall_topk_visits_freq(t1, 100)
        f2 = self.get_overall_topk_visits_freq(t2, 100)
        f1_dist, _ = EvalUtils.arr_to_distribution(f1, 0, 1, 100)
        f2_dist, _ = EvalUtils.arr_to_distribution(f2, 0, 1, 100)
        i_rank_jsd = EvalUtils.get_js_divergence(f1_dist, f2_dist)
        
        # grid mapping by number
        n_den1 = self.get_density(t1)                                               
        n_den2 = self.get_density(t2)
        n_den1_dist, _ = EvalUtils.arr_to_distribution(
            n_den1, 0, 1, 5000)
        n_den2_dist, _ = EvalUtils.arr_to_distribution(
            n_den2, 0, 1, 5000)
        number_density_jsd = EvalUtils.get_js_divergence(n_den1_dist, n_den2_dist)
        
        # grid mapping by size
        s_den1 = self.get_density(t1, True)
        s_den2 = self.get_density(t2, True)
        s_den1_dist, _ = EvalUtils.arr_to_distribution(
            s_den1, 0, 1, 5000)
        s_den2_dist, _ = EvalUtils.arr_to_distribution(
            s_den2, 0, 1,5000)
        size_density_jsd = EvalUtils.get_js_divergence(s_den1_dist, s_den2_dist)
        
        # # OD flow
        # od1 = self.get_odflow(t1)                                           
        # od2 = self.get_odflow(t2)
        # od1_dist, _ = EvalUtils.arr_to_distribution(
        #     od1, 0, 1, 10000)
        # od2_dist, _ = EvalUtils.arr_to_distribution(
        #     od2, 0, 1, 10000)
        # od_flow_jsd = EvalUtils.get_js_divergence(od1_dist, od2_dist)
        
        # Single O/D probability
        o_1, d_1 = self.get_single_odflow(t1)
        o_2, d_2 = self.get_single_odflow(t2)
        o_1_dist, _ = EvalUtils.arr_to_distribution(
            o_1, 0, 1, 10000)
        d_1_dist, _ = EvalUtils.arr_to_distribution(
            d_1, 0, 1, 10000)
        o_2_dist, _ = EvalUtils.arr_to_distribution(
            o_2, 0, 1, 10000)
        d_2_dist, _ = EvalUtils.arr_to_distribution(
            d_2, 0, 1, 10000)
        
        o_flow_jsd = EvalUtils.get_js_divergence(o_1_dist, o_2_dist)
        d_flow_jsd = EvalUtils.get_js_divergence(d_1_dist, d_2_dist)
        # p_od_flow_jsd = (o_flow_jsd + d_flow_jsd) / 2
        p_od_flow_jsd = d_flow_jsd
        
        # # gravity
        # gravity1 = self.get_gravity(t1)
        # gravity2 = self.get_gravity(t2)
        # gravity1_dist, _ = EvalUtils.arr_to_distribution(
        #     gravity1, 0, 1, 10000)
        # gravity2_dist, _ = EvalUtils.arr_to_distribution(
        #     gravity2, 0, 1, 10000)
        # gravity_flow_jsd = EvalUtils.get_js_divergence(gravity1_dist, gravity2_dist)
        gravity_flow_jsd = 0.
        
        # connectivity percentage
        con_perc, partial_con_perc = self.get_connectivity(t2)

        return distance_jsd, radius_jsd, location_jsd, number_density_jsd, size_density_jsd, p_od_flow_jsd, gravity_flow_jsd, g_rank_jsd, i_rank_jsd, con_perc, partial_con_perc




class CollectiveEval(object):
    """
    collective evaluation metrics
    """
    @staticmethod
    def get_visits(trajs, max_locs):
        """
        get probability distribution of visiting all locations
        :param trajs:
        :return:
        """
        visits = np.zeros(shape=(max_locs), dtype=float)
        for traj in trajs:
            for t in traj:
                visits[t] += 1
        visits = visits / np.sum(visits)
        return visits

    @staticmethod
    def get_topk_visits(visits, K):
        """
        get top-k visits and the corresponding locations
        :param trajs:
        :param K:
        :return:
        """
        locs_visits = [[i, visits[i]] for i in range(visits.shape[0])]
        locs_visits.sort(reverse=True, key=lambda d: d[1])
        topk_locs = [locs_visits[i][0] for i in range(K)]
        topk_probs = [locs_visits[i][1] for i in range(K)]
        return np.array(topk_probs), topk_locs

    @staticmethod
    def get_topk_accuracy(v1, v2, K):
        """
        get the accuracy of top-k visiting locations
        :param v1:
        :param v2:
        :param K:
        :return:
        """
        _, tl1 = CollectiveEval.get_topk_visits(v1, K)
        _, tl2 = CollectiveEval.get_topk_visits(v2, K)
        coml = set(tl1) & set(tl2)
        return len(coml) / K

    
def evaluate(path, max_len, boundary, grid_num, grid_size, filename, weighted):
    
    individualEval = IndividualEval(path, max_len, boundary, grid_num, grid_size, filename, weighted)
    
    if 'porto' in path:
        gene_trajs = read_data_from_file('save/porto/Seed/porto_steps=500_len=20_channel=256_bs=4096/logs/08-18-12-12-24_pre_40_best_270/gene_epoch_270.data')
    elif 'sz' in path:
        gene_trajs = read_data_from_file('save/sz/Seed/sz_steps=500_len=20_channel=128_bs=2048/logs/08-15-17-59-11_pre_60_best_360/gene_epoch_360.data')
    elif 'cd' in path:
        gene_trajs = read_data_from_file('save/cd/Seed/cd_steps=500_len=20_channel=256_bs=1024/logs/08-19-20-56-10_pre_3_best_24/gene_epoch_24.data')
    
    test_trajs = np.load(path+'/test.npy', allow_pickle=True).astype(np.int32)
    thres = min(len(gene_trajs), len(test_trajs))
    gene_trajs = gene_trajs[:thres]
    diversity = len(np.unique(gene_trajs, axis=0)) / len(gene_trajs)
    start_t = time.time()
    JSDs = individualEval.get_individual_jsds(test_trajs+1, gene_trajs)
    print('evaluation needs {}s'.format(time.time()-start_t))
    return JSDs, diversity
    



if __name__ == "__main__":
    # global
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset',default='sz',type=str)
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--grid_num', type=int, default=128)
    parser.add_argument('--grid_size', type=float, default=0.1)
    parser.add_argument('--filename', type=str, default='./node2vec/graph/{}.edgelist')
    parser.add_argument('--weighted', type=eval, choices=['True', 'False'], default=True)
    opt = parser.parse_args()

    set_seed()
    if opt.dataset == 'porto':
        boundary = {'min_lat': 41.147, 'max_lat': 41.178, 'min_lng': -8.65, 'max_lng': -8.53}
    elif opt.dataset == 'sz':
        boundary = {'min_lat': 22.48, 'max_lat': 22.58, 'min_lng': 113.9, 'max_lng': 114.1}
    elif opt.dataset == 'cd':
        boundary = {'min_lat': 30.6, 'max_lat': 30.75, 'min_lng': 104, 'max_lng': 104.16}
    else:
        raise ValueError('Unsupported dataset: {}'.format(opt.dataset))
    data_path = os.path.join(opt.data_dir, opt.dataset)
    weighted = opt.weighted
    filename = opt.filename.format(opt.dataset)
   
    
    epoch = 0
    JSDs, diversity = evaluate(data_path, opt.max_len, boundary, opt.grid_num, opt.grid_size, filename, weighted)
    distance_jsd, radius_jsd, location_jsd, number_density_jsd, size_density_jsd, p_od_flow_jsd, gravity_flow_jsd, g_rank_jsd, i_rank_jsd, con_perc, partial_con_perc = JSDs
    print('\nTest epoch:%d, Distance: %.4f, Radius: %.4f, Location: %.4f, NDensity: %.4f, SDensity: %.4f, ODProb: %.4f, Gravity: %.4f, G_rank: %.4f, I_rank: %.4f, Connectivity: %.4f, Partial Connectivity: %.4f, Diversity: %.4f'
                  % (epoch+1, distance_jsd, radius_jsd, location_jsd, number_density_jsd, size_density_jsd,  p_od_flow_jsd, gravity_flow_jsd, g_rank_jsd, i_rank_jsd, con_perc, partial_con_perc, diversity))