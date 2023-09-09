"""
Code within all the 'COMPLETE THIS METHOD/TODO' sections was written by Xingchen (Estella) Ye.
"""

import numpy as np
from hw5.utils import Node


class BayesNet:
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.topological_sort()
        self.set_children()
    
    def topological_sort(self):
        new = []
        while self.nodes:
            for n in self.nodes:
                if set(n.parents) <= set(new):
                    new.append(n)
                    self.nodes.remove(n)
        self.nodes = new

    def set_children(self):
        for n in self.nodes:
            for p in n.parents:
                p.children.add(n)


    """
    4.1 Generate sample and weight from Bayes net by likelihood weighting
    """        
    def weighted_sample(self, evidence: dict={}):
        """
        Args:
            evidence (Dict): {Node:0/1} mappings for evidence nodes.
        Returns:
            Dict: {Node:0/1} mappings for all nodes.
            Float: Sample weight. 
        """
        sample = {}
        weight = 1
        # TODO: 4.1
        for node in evidence:
            sample[node] = evidence[node]

        for node in self.nodes:
            parent_samples = []
            for idx, parent in enumerate(node.parents):
                parent_samples.append(sample[parent])

            if node in evidence:
                weight = weight * node.get_probs(parent_samples)[evidence[node]]
            else:
                sample[node] = node.sample(parent_samples)

        return sample, weight

    """
    4.2 Generate sample from Bayes net by Gibbs sampling
    """  
    def gibbs_sample(self, node: Node, sample: dict):
        """
        Args:
            node (Node): Node to resample.
            sample (Dict): {node:0/1} mappings for all nodes.
        Returns:
            Dict: {Node:0/1} mappings for all nodes.
        """ 
        new = sample.copy()
        # TODO: 4.2
        parent_evidence = []
        for parent in node.parents:
            parent_evidence.append(sample[parent])

        px_given_mb = [0, 0]

        for x in range(2):
            px_given_parents = node.get_probs(parent_evidence)[x]

            for child in node.children:
                c_parent_evidence = []
                for parent in child.parents:
                    if parent.name != node.name:
                        c_parent_evidence.append(sample[parent])
                    else:
                        c_parent_evidence.append(x)
                px_given_parents *= child.get_probs(c_parent_evidence)[sample[child]]
            
            px_given_mb[x] = px_given_parents

        px_given_mb = px_given_mb/np.sum(px_given_mb)
        new[node] = np.random.choice([0, 1], p=px_given_mb)

        return new

    """
    4.3 Generate a list of samples given evidence and estimate the distribution.
    """  
    def gen_samples(self, numSamples: int, evidence: dict={}, LW: bool=True):
        """
        Args:
            numSamples (int).
            evidence (Dict): {Node:0/1} mappings for evidence nodes.
            LW (bool): Use likelihood weighting if True, Gibbs sampling if False.
        Returns:
            List[Dict]: List of {node:0/1} mappings for all nodes.
            List[float]: List of corresponding weights.
        """       
        # TODO: 4.3

        samples = []
        weights = []

        if LW: 
            for i in range(numSamples):
                sample, w = self.weighted_sample(evidence)
                samples.append(sample)
                weights.append(w)
        
        else:
            
            count = 0 
            sample, _ = self.weighted_sample(evidence)
            while count < numSamples:
                for node in self.nodes:
                    if node not in evidence:
                        sample = self.gibbs_sample(node, sample)
                        count += 1
                        samples.append(sample)
                        weights.append(1)

                    if count >= numSamples:
                        break
        

        return samples, weights

    def estimate_dist(self, node: Node, samples: list[dict], weights: list[float]):
        """
        Args:
            node (Node): Node whose distribution we will estimate.
            samples (List[Dict]): List of {node:0/1} mappings for all nodes.
            weights (List[float]: List of corresponding weights.
        Returns:
            Tuple(float, float): Estimated distribution of node variable.
        """           
        # TODO: 4.3

        p_0_evidence = 0
        p_1_evidence = 0

        for i in range(len(weights)):
            sample = samples[i]
            weight = weights[i]
            if sample[node] == 0:
                p_0_evidence += weight
            else:
                p_1_evidence += weight

        sum = (p_0_evidence + p_1_evidence)
        p_0_evidence = p_0_evidence/sum
        p_1_evidence = p_1_evidence/sum

        return (p_0_evidence, p_1_evidence)
