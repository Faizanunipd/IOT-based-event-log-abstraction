#!/usr/bin/env python
# coding: utf-8

# ## import libraries

# In[1]:


import os
import time
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from datetime import datetime, timedelta

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances_argmin_min

import pm4py
from pm4py import read_xes
from pm4py.algo.evaluation import algorithm
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_algorithm
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to, remove_transition



# In[2]:


import warnings
warnings.filterwarnings('ignore')

# Set the maximum number of rows to display
pd.set_option('display.max_rows', 1000)  # Change 1000 to your desired value


# ## read dataset

# In[3]:

def hello():
    print('hello')


def get_log_stats(log):
    # event log columns
    print("\nEvent log have total {} columns and {} log activities".format(log.shape[1], log.shape[0]))
    print("======================================================")
    
    print("\nEvent Log have the follwing columns:")
    print("====================================")
    print(list(log.columns))
    

def read_log(xes_file_path):
    # Import the XES file as an event log
    event_log = read_xes(xes_file_path)
    return event_log


# ## split log

# In[4]:


def split_log(log, test_size=0.30, shuffle=False):
    train_log, test_log = train_test_split(log, test_size=test_size, random_state=42, shuffle=shuffle)
    # train_log, test_log = pm4py.split_train_test(log, 0.7)
    # train_log = pm4py.objects.conversion.log.variants.df_to_event_log_1v.apply(train_log)
    # test_log = pm4py.objects.conversion.log.variants.df_to_event_log_1v.apply(test_log)
    return train_log, test_log


# ## train and evaluate Petri Net

# In[5]:


def train_evaluate_petri_net(training_log, testing_log):
    net, im, fm = pm4py.discover_petri_net_inductive(training_log, noise_threshold=0.2)

    # q_o = algorithm.apply(testing_log, net, im, fm)
    # fitness = round(q_o['fitness']['average_trace_fitness'],3)

    # # log = pd.concat([training_log, testing_log])
    # q_o = algorithm.apply(training_log, net, im, fm)
    # prec = round(q_o['precision'],3)
    # gen = round(q_o['generalization'],3)
    # simp = round(q_o['simplicity'],3)

    # print("\n========================")
    # print("Fitness: ", fitness)
    # print("Precision: ", prec)
    # print("Generalization: ", gen)
    # print("Simplicity: ", simp)
    # print("========================\n\n")

    return net, im, fm


# ## save and plot models

# In[6]:


def __remove_silent_transitions(net):
    silent_transitions = [transition for transition in net.transitions if transition.label is None]

    # for t in silent_transitions:
    #     # Remove arcs connected to the invisible transition
    #     for arc in list(net.arcs):
    #         if arc.target == t or arc.source == t:
    #             net.arcs.remove(arc)
    #     # Remove the transition itself
    #     net.transitions.remove(t)

    for trans in silent_transitions:
        in_arcs = trans.in_arcs
        for arc in in_arcs:
            place = arc.source
            place.out_arcs.remove(arc)
            net.arcs.remove(arc)
        out_arcs = trans.out_arcs
        for arc in out_arcs:
            place = arc.target
            place.in_arcs.remove(arc)
            net.arcs.remove(arc)
        net.transitions.remove(trans)

    return net

def plot_petri_net(net, im, fm, remove_silent=False, save_graph=False, graph_name="Petri_Net.png"):
    gviz = pn_visualizer.apply(net, im, fm)
    pn_visualizer.view(gviz)

    net = __remove_silent_transitions(net) if remove_silent else net
        
    pn_visualizer.save(gviz, graph_name) if save_graph else None

def plot_bpmn(net, im, fm, save_graph=False, graph_name="BPMN.png"):
    bpmn_graph = pm4py.convert_to_bpmn(net, im, fm)

    gviz = bpmn_visualizer.apply(bpmn_graph)
    bpmn_visualizer.view(gviz)

    bpmn_visualizer.save(gviz, graph_name) if save_graph else None


# ## Create Session

# In[7]:


def log_preprocessing(event_log):
    event_log['time:timestamp'] = event_log['time:timestamp'].apply(lambda x : str(x))

    event_log['time:timestamp'] = event_log['time:timestamp'].apply(lambda x : x.split('.')[0])
    event_log['time:timestamp'] = event_log['time:timestamp'].apply(lambda x : x.split('+')[0])

    format_string = '%Y-%m-%d %H:%M:%S'
    event_log['time:timestamp'] = event_log['time:timestamp'].apply(lambda x : datetime.strptime(x, format_string))

    return event_log
    
def create_session(log, session_duration):
    event_log = log_preprocessing(log)
    
    customers = list(event_log['case:concept:name'].unique())
    if 'lifecycle:transition' in event_log.columns:
        session_df = pd.DataFrame(columns=['case:concept:name', 'time:timestamp', 'lifecycle:transition','concept:name',  'Session'])
    else:
        session_df = pd.DataFrame(columns=['case:concept:name', 'time:timestamp', 'concept:name',  'Session'])

    for customer in customers:
        customer_trace = event_log[event_log['case:concept:name'] == customer]
        customer_trace = customer_trace.sort_values('time:timestamp')
        customer_trace['Session'] = (customer_trace['time:timestamp'] - 
                                     customer_trace['time:timestamp'].shift(1)).gt(pd.Timedelta(minutes=session_duration)).cumsum() + 1
        session_df = pd.concat([session_df, customer_trace], ignore_index=True)

    return session_df


# ## Frequency Based Encoding

# In[8]:


def freq_encoding(session_log):
    # Perform one-hot encoding
    one_hot_encoded = pd.get_dummies(session_log['concept:name'], prefix='activity')
    
    # Replace frequency with 1 where frequency is not 0
    one_hot_encoded = one_hot_encoded.applymap(lambda x: 1 if x > 0 else 0)
    
    if 'lifecycle:transition' in session_log.columns:
        df_encoded = pd.concat([session_log[['case:concept:name', 'lifecycle:transition', 'Session']], one_hot_encoded], axis=1)
        df_grouped = df_encoded.groupby(['case:concept:name', 'lifecycle:transition', 'Session']).sum().reset_index()
    else:
        df_encoded = pd.concat([session_log[['case:concept:name', 'Session']], one_hot_encoded], axis=1)
        df_grouped = df_encoded.groupby(['case:concept:name', 'Session']).sum().reset_index()

    return df_grouped


def calculate_time_spent(records):
    time_spent_list = []
    for i in range(len(records)-1):
        current_record = records.iloc[i]
        next_record = records.iloc[i+1]

        current_time = current_record['time:timestamp']
        next_time = next_record['time:timestamp']

        if current_record['case:concept:name'] == next_record['case:concept:name'] and current_record['Session'] == next_record['Session']:
            time_spent = next_time - current_time
        else:
            time_spent = timedelta(minutes=15)

        time_spent_list.append(round(time_spent.total_seconds() / 60, 2))  # Convert to minutes

    # Adding time spent for the last record (assuming the last record is not considered in the loop)
    time_spent_list.append(15)

    return time_spent_list

def duration_encoding(session_log):
    time_spent = calculate_time_spent(session_log)
    session_log['duration'] = time_spent
    
    # Perform one-hot encoding
    one_hot_encoded = pd.get_dummies(session_log['concept:name'], prefix='activity')
    one_hot_encoded = one_hot_encoded.astype(float)
    
    # Concatenate one-hot encoded columns with original DataFrame
    if 'lifecycle:transition' in session_log.columns:
        df_encoded = pd.concat([session_log[['case:concept:name', 'concept:name', 'lifecycle:transition', 'Session', 'duration']], one_hot_encoded], axis=1)
    else:
        df_encoded = pd.concat([session_log[['case:concept:name', 'concept:name', 'Session', 'duration']], one_hot_encoded], axis=1)

    # Initialize new columns for each page name with default value 0
    for page_name in df_encoded['concept:name'].unique():
        df_encoded[page_name] = 0.0

    # Fill the one-hot encoded columns with the corresponding duration values
    for index, row in df_encoded.iterrows():
        page_name_col = row["concept:name"]
        df_encoded.at[index, page_name_col] = round(row['duration'], 2)

    # Drop unnecessary columns
    df_encoded = df_encoded.drop(['concept:name', 'duration'], axis=1)
    
    # Group by customer_id and session
    # df_encoded['time:timestamp'] = df_encoded['time:timestamp'].astype(str)
    if 'lifecycle:transition' in session_log.columns:
        df_grouped = df_encoded.groupby(['case:concept:name', 'lifecycle:transition', 'Session']).sum().reset_index()
    else:
        df_grouped = df_encoded.groupby(['case:concept:name', 'Session']).sum().reset_index()

    return df_grouped

# ## Clustering

# In[9]:

from sklearn.preprocessing import MinMaxScaler
def clustering_preprocessing(encoded_log):
    # Filter columns
    activity_columns = [col for col in encoded_log.columns if col.startswith('activity_')]
    
    # Select only the columns that start with 'activity_'
    features = encoded_log[activity_columns]

    # Standardize features
    # X = StandardScaler().fit_transform(features)
    # MinMax scaling
    scaler = MinMaxScaler()
    X = scaler.fit_transform(features)

    # Convert the transformed ndarray back to a DataFrame
    # You need to retain the column names and index of your original DataFrame
    transformed_df = pd.DataFrame(X, columns=features.columns, index=features.index)

    return transformed_df
    # return features


def elbow_clustering(encoded_log):
    features_log = clustering_preprocessing(encoded_log)
    
    # Compute K-Means for different values of k
    k_values = range(3, 21)  # Adjust the range of k as needed
    silhouette_scores = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(features_log)
        silhouette_scores.append(silhouette_score(features_log, labels))
    
    # Find the optimal k using the elbow method
    plt.plot(k_values, silhouette_scores, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    
    # Choose the best k based on the elbow point
    optimal_k = k_values[np.argmax(silhouette_scores)]
    print("Optimal number of clusters (k) based on the elbow method:", optimal_k)   

    return optimal_k

def estimate_eps(encoded_log):

    features_log = clustering_preprocessing(encoded_log)
    
    # Compute DBSCAN
    eps_values = np.linspace(0.1, 3.0, num=20)
    silhouette_scores = []
    
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps)
        labels = dbscan.fit_predict(features_log)
        silhouette_scores.append(silhouette_score(features_log, labels))
    
    # Plotting Elbow Method
    plt.plot(eps_values, silhouette_scores, marker='o')
    plt.xlabel('Eps')
    plt.ylabel('Silhouette Score')
    plt.title('DBSCAN Elbow Method')
    plt.grid(True)
    plt.show()


def KMeans_Clusteirng(encoded_log, k=2):

    features_log = clustering_preprocessing(encoded_log)

    # Apply DBSCAN clustering
    kmeans_model = KMeans(n_clusters=k, random_state=42 )  
    encoded_log['KMeans_Cluster'] = kmeans_model.fit_predict(features_log)

    labels = kmeans_model.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    print('\nEstimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    
    print("\n", encoded_log['KMeans_Cluster'].value_counts(), "\n")

    return encoded_log

def DBSACN_Clusteirng(encoded_log, epsilon_value=2, minimum_samples=12):

    features_log = clustering_preprocessing(encoded_log)


    # Calculate the percentage of zeros in each column
    zero_percentages = (features_log == 0).sum() / len(features_log)

    # Select columns where the percentage of zeros is less than or equal to 0.9 (90%)
    columns_to_keep = zero_percentages[zero_percentages <= 0.9].index

    # Filter the DataFrame to keep only selected columns
    transformed_df = features_log[columns_to_keep]

    # Apply DBSCAN clustering
    dbscan_model = DBSCAN(eps=epsilon_value, min_samples=minimum_samples)  # Adjust eps and min_samples based on your data
    encoded_log['DBSCAN_Cluster'] = dbscan_model.fit_predict(features_log)

    labels = dbscan_model.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    print('\nEstimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    
    # print("\n", encoded_log['DBSCAN_Cluster'].value_counts(), "\n")

    return encoded_log

def remove_noise_samples(encoded_log):
    # Assuming features_df is a DataFrame containing the feature vectors and DBSCAN_Cluster column
    # Reset the index to ensure proper alignment of indices
    # encoded_log = encoded_log.reset_index(drop=True)

    features_log = clustering_preprocessing(encoded_log)
    if 'DBSCAN_Cluster' in encoded_log.columns:
        cluster_column = 'DBSCAN_Cluster'
    else:
        cluster_column = 'KMeans_Cluster'

    features_log[cluster_column] = encoded_log[cluster_column]
    
    # Find noise indices
    noise_indices = encoded_log[encoded_log[cluster_column] == -1].index

    if len(noise_indices) >= 1:
    
        # Find the nearest cluster for each noise point
        nearest_cluster_indices, _ = pairwise_distances_argmin_min(
            features_log.iloc[noise_indices, :-1],  # Exclude the cluster label column
            features_log[features_log[cluster_column] != -1].iloc[:, :-1]  # Exclude noise points
        )
        
        # Merge noise samples into the nearest cluster
        encoded_log.loc[noise_indices, cluster_column] = encoded_log.loc[
            encoded_log[cluster_column] != -1, cluster_column
        ].iloc[nearest_cluster_indices].values
        
        # Check if there are still noise points
        remaining_noise_indices = encoded_log[encoded_log[cluster_column] == -1].index
        if len(remaining_noise_indices) > 0:
            print(f"T\nhere are still {len(remaining_noise_indices)} noise points remaining.\n")
        else:
            print("\nAll noise points have been assigned to the nearest cluster.\n")

        print("\n", encoded_log[cluster_column].value_counts(), "\n")
    return encoded_log


# ## Plotting

# In[10]:


def plot_heatmap2(encoded_log, largest_activities=20, mode="linear"):
    if 'DBSCAN_Cluster' in encoded_log.columns:
        cluster_column = 'DBSCAN_Cluster'
    else:
        cluster_column = 'KMeans_Cluster'

    features_log = clustering_preprocessing(encoded_log)
    features_log[cluster_column] = encoded_log[cluster_column]
    features_df = features_log
    
    # Filter numeric columns for plotting
    numeric_features = features_df.select_dtypes(include='number')
    
    # Calculate mean of each feature within each cluster
    cluster_means = numeric_features.groupby(cluster_column).mean().T
    
    # Calculate correlation matrix
    correlation_matrix = cluster_means.corr()
    
    # Filter top correlated features for each cluster from all features
    top_correlated_columns = {}
    for cluster_label in correlation_matrix.columns:
        top_correlated_columns[cluster_label] = correlation_matrix[cluster_label].nlargest(largest_activities).index.tolist()
    
    # Create a DataFrame containing only the top correlated columns for each cluster
    cluster_means_top_correlated = pd.DataFrame()
    for cluster_label, columns in top_correlated_columns.items():
        cluster_means_top_correlated[cluster_label] = cluster_means[columns].idxmax()
    
    # Plot heatmap
    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots()
    
    if mode == "linear":
        sns.heatmap(cluster_means_top_correlated, cmap="YlOrRd", annot=True, fmt="", cbar=False, linewidths=.5, xticklabels=True, yticklabels=True, ax=ax)
    else:
        sns.heatmap(cluster_means_top_correlated, cmap="YlOrRd", annot=True, fmt="", cbar=False, linewidths=.5, norm=LogNorm(), xticklabels=True, yticklabels=True, ax=ax)
    
    plt.title('Heatmap of Top {} Most Correlated Columns with Each Feature'.format(largest_activities))
    plt.xlabel('Cluster')
    plt.ylabel('Feature')
    plt.show()
    
    # Return the most correlated feature names for each cluster
    most_correlated_features = {}
    for cluster_label, columns in top_correlated_columns.items():
        most_correlated_features[cluster_label] = numeric_features.columns[cluster_means[columns].idxmax()]
    
    return most_correlated_features


def plot_heatmap(encoded_log, largest_activities=20, mode="linear"):
    if 'DBSCAN_Cluster' in encoded_log.columns:
        cluster_column = 'DBSCAN_Cluster'
    else:
        cluster_column = 'KMeans_Cluster'

    features_log = clustering_preprocessing(encoded_log)
    features_log[cluster_column] = encoded_log[cluster_column]
    features_df = features_log
    
    # Calculate mean of each feature within each cluster
    cluster_means = (features_df.groupby(cluster_column).mean()).T
    
    # Calculate correlation matrix
    correlation_matrix = cluster_means.corr()
    

    # Select top 10 most correlated columns for each cluster
    top_correlated_columns = {}
    for cluster_label in correlation_matrix.columns:
        top_correlated_columns[cluster_label] = correlation_matrix[cluster_label].nlargest(largest_activities).index.tolist()
    
    # Create a DataFrame containing only the top 10 most correlated columns for each cluster
    cluster_means_top_correlated = pd.DataFrame()
    for cluster_label, columns in top_correlated_columns.items():
        cluster_means_top_correlated[cluster_label] = cluster_means[cluster_label][columns]
    
    # Plot heatmap
    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots()
    logmin = 0.001
    # sns.heatmap(cluster_means_top_correlated, annot=True, cmap="hot", fmt=".2f", facecolor='white')
    if mode == "linear":
        sns.heatmap(cluster_means_top_correlated, cmap="YlOrRd", linewidths=.5,xticklabels=True, yticklabels= True, ax = ax)
    else:
        sns.heatmap(cluster_means_top_correlated, cmap="YlOrRd", linewidths=.5,norm =LogNorm(), xticklabels=True, yticklabels= True, ax= ax)
    
    plt.title('Heatmap of Top {} Most Correlated Columns with Each Feature')
    plt.xlabel('Cluster')
    plt.ylabel('Feature')
    plt.show()

    cluster_map = {}
    for col in cluster_means_top_correlated.columns:
        sorted_values = cluster_means_top_correlated[col].sort_values(ascending=False)
        highest_value = sorted_values.iloc[0]
        highest_index = sorted_values.index[0]
        # second_highest_value = sorted_values.iloc[1]
        # second_highest_index = sorted_values.index[1]
        # cluster_map[col] = {
        #     'highest_value': highest_value,
        #     'highest_index': highest_index,
        #     'second_highest_value': second_highest_value,
        #     'second_highest_index': second_highest_index
        # }

        cluster_map[col] = highest_index
    
    activity_dict_no_activity = {k: v.replace('activity_', '') for k, v in cluster_map.items()}

    return activity_dict_no_activity
 

def calcCenters(enc):
    if 'DBSCAN_Cluster' in enc.columns:
        cluster_column = 'DBSCAN_Cluster'
    else:
        cluster_column = 'KMeans_Cluster'
        
    y_pred = list(enc[cluster_column])
    enc = clustering_preprocessing(enc)

    centers = [[] for i in range(max(y_pred)+1)]
    for v in range(max(y_pred)+1):
        cluster = []
        indexPos = [ i for i in range(len(y_pred)) if y_pred[i] == v ]
        for i in indexPos :
            cluster.append(enc.iloc[i])
        centers[v] = np.mean(cluster, axis=0)
    return centers


def betterPlotting (distinct,centers, enc, path, params, mode = "linear"):
    if 'DBSCAN_Cluster' in enc.columns:
        cluster_column = 'DBSCAN_Cluster'
    else:
        cluster_column = 'KMeans_Cluster'
        
    y_pred = list(enc[cluster_column])

    if (len(centers[0])> len(distinct)):
        attr = [c[len(distinct):] for c in centers]
        attrNames = []
    else:
        attr = []
        attrNames = []
    centers = np.array([c[range(len(distinct))] for c in centers])
    
    #Normalizzazione solo per plotting
    if any(i>1 for c in centers for i in c):
        for i,c in enumerate(centers):
            lower =  min(c)
            if lower <0 :
                lower = abs(lower)
                centers[i] = [j+lower for j in c]
        centers = centers/centers.sum(axis=1,keepdims = True)
    newCenters= []
    newDistinct = []
    for i,e in enumerate(distinct):
        drop = True
        for c in centers:
            if c[i] >= 0.01: drop = False
        if not drop: newDistinct.append(e)
    for i,c in enumerate(centers):
        cn = []
        for e in newDistinct:
            cn.append(c[distinct.index(e)])
        if attr != []: cn = [*cn, *attr[i]]
        newCenters.append(cn)
    if attr != []: columns = newDistinct + attrNames
    else: columns = newDistinct
    df1 = pd.DataFrame(newCenters,index=range(max(y_pred)+1),columns =columns)
    logmin = 0.001
    fig, ax = plt.subplots()
    fig.set_size_inches((len(columns),len(newCenters)))
    
    if mode == "linear":
        sns.heatmap(df1, cmap="YlOrRd", linewidths=.5,xticklabels=True, yticklabels= True, ax = ax)
    else:
        sns.heatmap(df1, cmap="YlOrRd", linewidths=.5,norm =LogNorm(), vmin= max(centers.min().min(),logmin),xticklabels=True, yticklabels= True, ax= ax)
    if attr != []:
        ax.vlines([len(newDistinct)], *ax.get_ylim())
    # params['threshold (min)'] = self.threshold_value
    ax.set_title(params)
    fig.savefig(path+".png",bbox_inches='tight') 
    # ax.clear()
    return df1


# Function to get maximum value column name for each row
def max_value_column(row):
    max_col = row.idxmax()
    return {row.name: max_col}


def get_cluster_map(df1):
    # Apply the function row-wise
    result = df1.apply(max_value_column, axis=1)
    
    # Convert the result to a dictionary
    cluster_map = {}
    for item in result:
        cluster_map.update(item)

    return cluster_map

# ## Abstract Model

# In[11]:


# Define a function to label the activities as 'start' and 'end'
def __label_activity(group):
    # print(group)
    if len(group) == 1:
        group.loc[group.index[0], 'Activity'] = 'start'
        new_activity = group.iloc[0].copy()  # Copy the first activity
        new_activity['Activity'] = 'end'
        return pd.concat([group, pd.DataFrame([new_activity])], ignore_index=True)
    else:
        group.loc[group.index[0], 'Activity_Status'] = 'start'
        group.loc[group.index[-1], 'Activity_Status'] = 'end'
        return group

def abstract_log_preprocessing(session_log, cluster_log, cluster_map):
    if 'DBSCAN_Cluster' in cluster_log.columns:
        cluster_column = 'DBSCAN_Cluster'
    else:
        cluster_column = 'KMeans_Cluster'

    # Perform inner join
    merged_log = pd.merge(session_log, cluster_log, on=['case:concept:name', 'Session', 'lifecycle:transition'], how='inner')
    
    merged_log = merged_log[['case:concept:name', 'Session', 'lifecycle:transition', 'time:timestamp', 'concept:name', cluster_column]]
    
    merged_log[cluster_column] = merged_log[cluster_column].replace(cluster_map)
    merged_log.rename({'concept:name':'log_activity', cluster_column:'abstract_activity'}, axis=1, inplace=True)

    merged_log = merged_log.sort_values(by=['case:concept:name', 'Session', 'time:timestamp'])
    
    # Group the DataFrame by 'CustomerID' and 'Session', then apply the function
    new_df = merged_log.groupby(['case:concept:name', 'Session']).apply(__label_activity).reset_index(drop=True)
    
    # Select only the 'start' and 'end' activities
    merged_log = new_df[new_df['Activity_Status'].isin(['start', 'end'])]

    activity_log = merged_log[['case:concept:name', 'time:timestamp','log_activity', 'abstract_activity', 'Activity_Status']]
    activity_log.rename({'log_activity':'concept:name'}, axis=1, inplace=True)
    
    abstract_log = merged_log[['case:concept:name', 'time:timestamp','abstract_activity', 'Activity_Status']]
    abstract_log.rename({'abstract_activity':'concept:name'}, axis=1, inplace=True)

    return activity_log, abstract_log


# In[47]:


def add_invisible_transitions(net, im, fm, in_place, out_place):
    ip = [place for place in im][0]
    fp = [place for place in fm][0]

    transition = PetriNet.Transition("tau")
    net.transitions.add(transition)
    add_arc_from_to(transition, ip, net)
    add_arc_from_to(in_place, transition, net)
    
    transition = PetriNet.Transition("tau")
    net.transitions.add(transition)
    add_arc_from_to(fp, transition, net)
    add_arc_from_to(transition, out_place, net)


def merge_subprocesses(train_abstract_log, train_activity_log, test_activity_log):
    
    noise_threshold = 0.2
    
    print(f'Abstracted model is started to create at {datetime.fromtimestamp(time.time())}.')
    
    train_abstract_log = train_abstract_log[['case:concept:name','time:timestamp','concept:name']]
    a_net, a_im, a_fm = pm4py.discover_petri_net_inductive(train_abstract_log, noise_threshold)
        
    pm4py.write_pnml(a_net, a_im, a_fm, "_AbstractedModel.pnml")
    pm4py.save_vis_petri_net(a_net, a_im, a_fm, "_AbstractedModel.pdf")
    
    print(f'\nSubprocesses are started to merge at {datetime.fromtimestamp(time.time())}')
    
    low_level_activities = {t for t in a_net.transitions if t.label is not None}
    for i, a in enumerate(low_level_activities):
        print(f"Cluster {i}, High level activity: {a.label}  is merging at {datetime.fromtimestamp(time.time())}.")
        for arc in a.in_arcs:
            in_place = arc.source
        for arc in a.out_arcs:
            out_place = arc.target
        
        cluster_number = a.label[-1]
        sub_log = train_activity_log[train_activity_log['abstract_activity'] == a.label]
        train_activity_log_sub = sub_log[['case:concept:name','time:timestamp','concept:name', 'abstract_activity']]

        net, im, fm = pm4py.discover_petri_net_inductive(train_activity_log_sub, noise_threshold)
        add_invisible_transitions(net, im, fm, in_place, out_place)
        
        a_net.transitions.update(net.transitions)
        a_net.places.update(net.places)
        a_net.arcs.update(net.arcs)
        remove_transition(a_net, a)
        
    # pm4py.write_pnml(a_net, a_im, a_fm, os.path.join(abstracted_logs_folder, str(original_log_name+"_MergedModel.pnml")))
    # pm4py.save_vis_petri_net(a_net, a_im, a_fm, os.path.join(abstracted_logs_folder, str(original_log_name+"_MergedModel.pdf")))

    # q_o = algorithm.apply(test_activity_log, a_net, a_im, a_fm)
    # fitness = round(q_o['fitness']['average_trace_fitness'],3)
    q_o = algorithm.apply(train_activity_log, a_net, a_im, a_fm)
    fitness = round(q_o['fitness']['average_trace_fitness'],3)
    prec = q_o['precision']
    gen = round(q_o['generalization'],3)
    simp = round(q_o['simplicity'],3)
    
    # fitness = pm4py.fitness_token_based_replay(testing_log, net, im, fm)
    # prec = pm4py.precision_token_based_replay(testing_log, net, im, fm)
    # gen = generalization_evaluator.apply(testing_log, net, im, fm)
    # simp = simplicity_evaluator.apply(net)
    
    print("\n========================")
    print("Fitness: ", fitness)
    print("Precision: ", prec)
    print("Generalization: ", gen)
    print("Simplicity: ", simp)
    print("========================\n")

    return a_net, a_im, a_fm


def distance (center, enc):
    difference = np.diff([center,enc], axis = 0)
    return np.sqrt(np.sum(np.power(difference,2)))


def epsEstimate (enc,imgpath=""):
    enc = clustering_preprocessing(enc)

    enc = enc.values.tolist()
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(enc)
    _,indices = nbrs.kneighbors(enc)
    eps_dist = []
    for i,e in enumerate(enc): 
        eps_dist.append(distance(e,enc[indices[i][-1]]))
    eps_dist.sort()
    
    tot = len(eps_dist)
    fig,ax1 = plt.subplots(figsize=(10,5))
    hist, bins, _ = ax1.hist(eps_dist[:-math.floor(tot/5)],bins= 50)
    perc = [0 for i in bins]

    cumsum = np.cumsum(hist)
    percent = (cumsum / tot) * 100
    
    # Find the knee of the curve where the percentage plateaus
    knee_index = np.argmax(np.gradient(percent))
    best_eps = bins[knee_index]

    for i,v in enumerate(bins):
        for e in eps_dist:
            if e<v:
                perc[i]+=1
    for i,_ in enumerate(perc):
        perc[i] = (perc[i]/tot)*100
    ax1.set_xticks(bins)
    ax1.xaxis.set_tick_params(rotation=90)
    ax1.set_title("Eps Estimate ")
    ax1.set_ylabel('n')
    ax1.set_xlabel('Eps')
    ax2 = ax1.twinx()
    ax2.plot(bins,perc,color= 'red')
    ax2.set_yticks(np.linspace(0,100,num = 11))
    ax2.set_ylabel('percentage')
    ax1.grid(color= 'C0',alpha = 0.5)
    ax2.grid(color= 'red',alpha = 0.5)
    ax2.hlines(50,xmin=min(bins),xmax = max(bins),linestyle = 'dashed',color = 'k')
    ax2.tick_params(axis='y', colors='red')
    ax1.tick_params(axis='y', colors='C0')
    fig.tight_layout()
    fig.savefig("epsEstimate.png")
    fig.clear()

    return best_eps


def minPointsEstimate (enc, eps, imgpath=""):
    enc = clustering_preprocessing(enc)
    tree = BallTree(np.array(enc))
    allNgbr = []
    allNgbr.append(tree.query_radius(enc, eps, count_only=True))
    _, bins, _ = plt.hist (allNgbr, bins = 45)
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(bins, rotation = 90 )
    plt.title("MinPts Estimate")
    plt.ylabel('Number of sessions')
    plt.xlabel('Number of neighbors')
    plt.tight_layout()
    plt.savefig("minptsEstimate.png")
    plt.close()

    # Determine the best value for min_samples
    min_samples = int(np.ceil(np.mean(allNgbr)))
    return min_samples


def alignmennt_base_fitness(test_abs_log, net, im, fm):
    # Assuming test_abs_log is your event log and net is your Petri net
    alignments = alignments_algorithm.apply(test_abs_log, net, im, fm)
    
    total_fitness = 0
    num_alignments = len(alignments)
    
    for alignment in alignments:
        total_fitness += alignment['fitness']
    
    if num_alignments > 0:
        average_fitness = total_fitness / num_alignments
    else:
        average_fitness = 0  # Avoid division by zero if there are no alignments
    
    fitness = round(average_fitness, 3)
    return fitness


# Step 1: Train an inductive miner on high-level activities (clusters)
def train_inductive_miner(log):
    # log = pm4py.convert_to_event_log(log)
    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log, noise_threshold=0.2)
    # f.plot_petri_net(net, initial_marking, final_marking)
    return net, initial_marking, final_marking


# Step 2: Train an inductive miner on low-level activities for each cluster
def train_inductive_miners_for_clusters(log, clusters):
    miners = {}
    for cluster in clusters:
        filtered_log = log[log['abstract_activity'] == cluster]
        net, initial_marking, final_marking = train_inductive_miner(filtered_log)
        print("Cluster: ", cluster)
        plot_bpmn(net, initial_marking, final_marking)
        miners[cluster] = (net, initial_marking, final_marking)
    print("End Individual Models")
    return miners


def replace_transitions_with_miners(abstract_net, miners):
    for cluster, (net, _, _) in miners.items():
        # Find transitions in the abstract net that correspond to the current cluster
        cluster_transitions = [t for t in abstract_net.transitions if t.label == cluster]
        if len(cluster_transitions) == 0:
            continue
        
        abstract_net.transitions.update(net.transitions)
        abstract_net.places.update(net.places)
        abstract_net.arcs.update(net.arcs)
        remove_transition(abstract_net, cluster)
    
    return abstract_net



def get_fScore(fitness, precision):
    f_score = (2*fitness*precision)/(fitness+precision)
    return f_score

def evaluate_model(training_log, testing_log, net, im, fm, abstraction=False):

    if abstraction:
        fitness = alignmennt_base_fitness(testing_log, net, im, fm)
    else:
        q_o = algorithm.apply(testing_log, net, im, fm)
        fitness = round(q_o['fitness']['average_trace_fitness'],3)

    # Concatenate the DataFrames along rows
    log = pd.concat([training_log, testing_log], ignore_index=True)

    q_o = algorithm.apply(log, net, im, fm)
    prec = round(q_o['precision'],3)
    gen = round(q_o['generalization'],3)
    simp = round(q_o['simplicity'],3)
    f_score = get_fScore(fitness, prec)

    print("\nEvaluation Scores:")
    print("=====================")
    print("Fitness: ", fitness)
    print("Precision: ", prec)
    print("Generalization: ", gen)
    print("Simplicity: ", simp)
    print("\n\n")

    return fitness, prec, gen, simp, f_score


def thresholdView (event_log):
    difference = []
    for caseid,case in enumerate(event_log):
        added = []
        for event_id, event in enumerate(case):
            if event['lifecycle:transition'] == 'start' and event_id>0:
                difference.append ((event["time:timestamp"]-next((e['time:timestamp'] for e in reversed(case[:event_id]) if e['lifecycle:transition'] == 'start'),event["time:timestamp"])).total_seconds())
                ec = next((e for e in case[event_id:] if e['concept:name'] == event['concept:name'] and e['lifecycle:transition'] == 'complete'),None)
                if ec is not None and not started(ec, case, event_id):
                    added.append(index_event(ec, case))
            elif event['lifecycle:transition'] == 'complete' and event_id>0 and event_id not in added:
                difference.append((event["time:timestamp"]-next((e['time:timestamp'] for e in reversed(case[:event_id]) if e['lifecycle:transition'] == 'start'),case[-1]['time:timestamp'])).total_seconds())
            if event_id>0 and event_id not in added:
                difference.append((event["time:timestamp"]-case[-1]['time:timestamp']).total_seconds())
                
    if difference != []:
        # mean = np.mean([d for d in difference if d>0])
        # median = st.median([d for d in difference if d>0])
        mean = abs(np.mean([d for d in difference]))
        median = abs(np.median([d for d in difference]))
        if mean > median:
            # print('Threshold mean',mean/60, ' min')
            threshold_value = mean
        else: 
            # print('Threshold median',median/60, ' min')
            threshold_value = median
    #     plt.plot(difference,marker='o',linewidth = 0,zorder=1)
    #     plt.hlines(mean,xmin = 0, xmax = len(difference),color='red',linewidth=2,zorder = 2,label= 'Mean value = '+str(mean/60)+' min')
    #     plt.hlines(median,xmin = 0, xmax = len(difference),color='orange',linewidth=2,zorder = 3,label= 'Median value = '+str(median/60)+' min')
    #     plt.legend()
    #     plt.savefig('thresholdView.png')
    #     plt.close()
    #     print('Threshold Variance' , np.std(np.divide(difference,[60 for i in difference])))
    return threshold_value, difference

def started (event, case, id):
    # print("Event in case:", event in case)  # Should print True if event is in case
    # print("Type of event:", type(event))    # Helps to understand what type event is
    # ind = case.index(event)

    # Directly use identity comparison to find the index
    for index, existing_event in enumerate(case):
        if existing_event is event:
            ind = index
            break

    starter = next((e for e in case[id+1:ind] if e['lifecycle:transition']== 'start' and e['concept:name']==event['concept:name']),None)
    if starter is not None:
        return True
    else:
        return False
    

def index_event(event, case):
     # Directly use identity comparison to find the index
    for index, existing_event in enumerate(case):
        if existing_event is event:
            return index

    return False
    

import statistics as st
# def thresholdView2(event_log):
#     difference = []
#     for caseid,case in enumerate(event_log):
#         added = []
#         for event_id, event in enumerate(case):
#             if event['lifecycle:transition'] == 'start' and event_id>0:
#                 difference.append ((event["time:timestamp"]-next((e['time:timestamp'] for e in reversed(case[:event_id]) if e['lifecycle:transition'] == 'start'),event["time:timestamp"])).total_seconds())
#                 ec = next((e for e in case[event_id:] if e['concept:name'] == event['concept:name'] and e['lifecycle:transition'] == 'complete'),None)
#                 if ec is not None and not started(ec, case, event_id):
#                     added.append(case.index(ec))
#             elif event['lifecycle:transition'] == 'complete' and event_id>0 and event not in added:
#                 difference.append((event["time:timestamp"]-next((e['time:timestamp'] for e in reversed(case[:event_id]) if e['lifecycle:transition'] == 'start'),case[-1]['time:timestamp'])).total_seconds())
#     if difference != []:
#         mean = np.mean([d for d in difference if d>0])
#         median = st.median([d for d in difference if d>0])
#         print('mean',mean)
#         plt.plot(difference,marker='o',linewidth = 0,zorder=1)
#         plt.hlines(mean,xmin = 0, xmax = len(difference),color='red',linewidth=2,zorder = 2,label= 'Mean value = '+str(mean/60)+' min')
#         plt.hlines(median,xmin = 0, xmax = len(difference),color='orange',linewidth=2,zorder = 3,label= 'Median value = '+str(median/60)+' min')
#         plt.legend()
#         plt.savefig('thresholdView.png')
#         plt.close()
#         print('THRESHOLD VARIANCE' , np.std(np.divide(difference,[60 for i in difference])))


def estimate_thresold(event_log):
    event_log = pm4py.convert_to_event_log(event_log)
    threshold_value, difference = thresholdView(event_log)

    difference_abs = np.abs(difference)
    difference_abs = difference_abs / 60
    median = round(np.median(difference_abs))
    
    # Plot the boxplot
    plt.figure(figsize=(5, 1))
    plt.boxplot(difference_abs, vert=False, showfliers=False, widths=0.7)
    plt.title('Boxplot of Time Differences')
    plt.xlabel('Time Differences')
    plt.yticks([])
    # plt.ylabel('Minutes')
    plt.grid(False)
    plt.savefig("thresholdView.png")
    plt.show()
    # plt.close()

    return(median)