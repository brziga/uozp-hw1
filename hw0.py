from math import sqrt

NAN = float("nan")

def manhattan_dist(r1, r2):
    """ Arguments r1 and r2 are lists of numbers """
    distance = 0
    allCount = len(r1) #we assume both lists given are of equal length
    includedCount = 0
    for i in range(allCount):
        if r1[i] == r1[i] and r2[i] == r2[i]:
            includedCount += 1
            distance += abs(r1[i] - r2[i])
    return distance * (allCount/includedCount) if includedCount != 0 else NAN


def euclidean_dist(r1, r2):
    distance = 0
    allCount = len(r1) #we assume both lists given are of equal length
    includedCount = 0
    for i in range(allCount):
        if r1[i] == r1[i] and r2[i] == r2[i]:
            includedCount += 1
            distance += (r1[i] - r2[i])**2
    return sqrt(distance * (allCount/includedCount)) if includedCount != 0 else NAN


def single_linkage(c1, c2, distance_fn):
    """ Arguments c1 and c2 are lists of lists of numbers
    (lists of input vectors or rows).
    Argument distance_fn is a function that can compute
    a distance between two vectors (like manhattan_dist)."""
    distances = []
    for iC1 in range(len(c1)):
        for iC2 in range(len(c2)):
            distance = distance_fn(c1[iC1], c2[iC2])
            if distance == distance:
                #is a number
                distances.append(distance)
    return min(distances) if len(distances) > 0 else NAN



def complete_linkage(c1, c2, distance_fn):
    distances = []
    for iC1 in range(len(c1)):
        for iC2 in range(len(c2)):
            distance = distance_fn(c1[iC1], c2[iC2])
            if distance == distance:
                #is a number
                distances.append(distance)
    return max(distances) if len(distances) > 0 else NAN


def average_linkage(c1, c2, distance_fn):
    distances = []
    for iC1 in range(len(c1)):
        for iC2 in range(len(c2)):
            distance = distance_fn(c1[iC1], c2[iC2])
            if distance == distance:
                #is a number
                distances.append(distance)
    return sum(distances)/len(distances) if len(distances) > 0 else NAN


def flatten(lst):
    # this function was written by ChatGPT
    flattened_list = []
    for item in lst:
        if isinstance(item, list):
            flattened_list.extend(flatten(item))
        else:
            flattened_list.append(item)
    return flattened_list


class HierarchicalClustering:

    def __init__(self, cluster_dist, return_distances=False):
        # the function that measures distances clusters (lists of data vectors)
        self.cluster_dist = cluster_dist

        # if the results of run() also needs to include distances;
        # if true, each joined pair in also described by a distance.
        self.return_distances = return_distances

    def closest_clusters(self, data, clusters):
        """
        Return the closest pair of clusters and their distance.
        """
        closestClusters = [None, None, None] # [distance, cluster1, cluster2]
        for c1 in range(len(clusters)):
            for c2 in range(len(clusters)):
                if c1 == c2: continue # skip when both indexes point to the same element
                
                clustersData = [[], []]
                for c in (c1, c2):
                    for i in clusters[c]:
                        # element = i[0] if isinstance(i, list) else i
                        if isinstance(i, list):
                            pass
                        
                        if isinstance(i, list):
                            element = flatten(i)
                        else:
                            element = [i]
                        for x in element:
                            if isinstance(x, float): continue
                            clustersData[0 if c==c1 else 1].append(data[x])
                        # extractedData = data[element]
                        # for edel in extractedData:
                        #     clustersData[0 if c==c1 else 1].append(edel)

                distance = self.cluster_dist(clustersData[0], clustersData[1])
                if distance != distance: continue # skip as per instructions
                if closestClusters[0] is None or distance < closestClusters[0]:
                    # new closest pair has been found
                    closestClusters = [distance, c1, c2]
        return (clusters[closestClusters[1]], clusters[closestClusters[2]], closestClusters[0])

    def run(self, data):
        """
        Performs hierarchical clustering until there is only a single cluster left
        and return a recursive structure of clusters.
        """

        # clusters stores current clustering. It starts as a list of lists
        # of single elements, but then evolves into lists like
        # [[["Albert"], [["Branka"], ["Cene"]]], [["Nika"], ["Polona"]]]
        clusters = [[name] for name in data.keys()]

        while len(clusters) >= 2:
            first, second, distance = self.closest_clusters(data, clusters)
            # update the "clusters" variable
            
            firstIndex, secondIndex = clusters.index(first), clusters.index(second)
            if self.return_distances:
                newClusters = [ [clusters[firstIndex], clusters[secondIndex], distance] ]
            else:
                newClusters = [ [clusters[firstIndex], clusters[secondIndex]] ]
            for cluster in clusters:
                if cluster == first or cluster == second: continue
                newClusters.append(cluster)
            clusters = newClusters

        return clusters


def silhouette(el, clusters, data):
    """
    Za element el ob podanih podatkih data (slovar vektorjev) in skupinah
    (seznam seznamov nizov: ključev v slovarju data) vrni silhueto za element el.
    """
    distanceFunc = euclidean_dist
    linkageFunc = average_linkage

    # find out what cluster contains the element
    belongId = None
    for i in range(len(clusters)):
        if el in clusters[i]: 
            belongId = i
            break
    
    elData = data[el]

    # calc a
    aDistances = []
    for clusterEl in clusters[belongId]:
        if clusterEl == el: continue
        ceData = data[clusterEl]
        aDistances.append(distanceFunc(elData, ceData))
    a = sum(aDistances)/len(aDistances) if len(aDistances)>0 else 0
    if a == 0: return 0 # thanks https://en.wikipedia.org/wiki/Silhouette_(clustering)#:~:text=Note%20that%20a(i)%20is%20not%20clearly%20defined%20for%20clusters%20with%20size%20%3D%201%2C%20in%20which%20case%20we%20set

    #calc b
    #   find closest cluster
    closestClusterId = None
    closestClusterDist = None
    for i in range(len(clusters)):
        if i == belongId: continue

        belongCluster = []
        for index in clusters[belongId]:
            belongCluster.append(data[index])

        compareCluster = []
        for index in clusters[i]:
            compareCluster.append(data[index])

        distance = linkageFunc(belongCluster, compareCluster, distanceFunc)
        if closestClusterDist is None or distance < closestClusterDist:
            closestClusterDist = distance
            closestClusterId = i
    
    bDistances = []
    for clusterEl in clusters[closestClusterId]:
        ceData = data[clusterEl]
        bDistances.append(distanceFunc(elData, ceData))
    b = sum(bDistances)/len(bDistances)

    s = (b - a) / max(a, b) if max(a, b) != 0 else NAN
    return s


def silhouette_average(data, clusters):
    """
    Za podane podatke (slovar vektorjev) in skupine (seznam seznamov nizov:
    ključev v slovarju data) vrni povprečno silhueto.
    """
    dataKeys = list(data.keys())
    silhs = []
    for key in dataKeys:
        silhs.append(silhouette(key, clusters, data))
    avgS = sum(silhs)/len(silhs)
    
    return avgS


if __name__ == "__main__":

    data = {"a": [1, 2],
            "b": [2, 3],
            "c": [5, 5]}

    def average_linkage_w_manhattan(c1, c2):
        return average_linkage(c1, c2, manhattan_dist)

    hc = HierarchicalClustering(cluster_dist=average_linkage_w_manhattan)
    clusters = hc.run(data)
    print(clusters)  # [[['c'], [['a'], ['b']]]] (or equivalent)

    hc = HierarchicalClustering(cluster_dist=average_linkage_w_manhattan,
                                return_distances=True)
    clusters = hc.run(data)
    print(clusters)  # [[['c'], [['a'], ['b'], 2.0], 6.0]] (or equivalent)
