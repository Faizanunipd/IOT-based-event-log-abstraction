import os
from datetime import datetime

import pm4py
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

import functions as f 


def flat_model(event_log):
    event_log = event_log[['case:concept:name', 'time:timestamp', 'lifecycle:transition', 'concept:name']]

    train_log, test_log = f.split_log(event_log)
    net, im, fm = f.train_evaluate_petri_net(train_log, test_log)
    fitness, prec, gen, simp, f_score = f.evaluate_model(train_log, test_log, net, im, fm)

    f.plot_petri_net(net, im, fm)
    f.plot_bpmn(net, im, fm)

    # Convert DataFrame to pm4py event log
    event_log_converted = pm4py.convert_to_event_log(event_log, case_id_key="case:concept:name")

    # Export event log to XES format
    pm4py.write_pnml(net, im, fm, "_FlatModel.pnml")
    xes_exporter.apply(event_log_converted, "_ActivityLog.xes")
    return ['Flat', '#', '#', fitness, prec, gen, simp, f_score]


def clustering(event_log, params):
    encoding = params.get('encoding')
    session_duration = params.get('session_duration')
    clustering_tech = params.get('clustering')
    print(clustering_tech)
    eps = params.get('eps')
    minSample = params.get('minSample')
    k = params.get('k')

    if encoding == "freq":
        session_log = f.create_session(event_log, session_duration)
        encoded_log = f.freq_encoding(session_log)
    else:
        session_log = f.create_session(event_log, session_duration)
        encoded_log = f.duration_encoding(session_log)

    if clustering_tech == "DBSCAN":
        encoded_log = f.DBSACN_Clusteirng(encoded_log, eps, minSample)
    else:
        k = f.elbow_clustering(encoded_log)
        encoded_log = f.KMeans_Clusteirng(encoded_log, k)

    encoded_log = f.remove_noise_samples(encoded_log)
    
    centers = f.calcCenters(encoded_log)
    distinct = list(event_log['concept:name'].unique())

    current_date = datetime.now().strftime("%Y%m%d")
    params = {"alg":"dbscan", "assignNoisy" : True}
    df1 = f.betterPlotting(distinct, centers,encoded_log,f"heatmap_{current_date}",params)

    cluster_map = f.get_cluster_map(df1)
    return session_log, encoded_log, cluster_map


def abstract_model(session_log, encoded_log, cluster_map, params):
    encoding = params.get('encoding')
    clustering_tech = params.get('clustering')
    activity_log, abstract_log = f.abstract_log_preprocessing(session_log, encoded_log, cluster_map)

    train_abs_log, test_abs_log = f.split_log(abstract_log)
    net, im, fm = f.train_evaluate_petri_net(train_abs_log, test_abs_log)
    fitness, prec, gen, simp, f_score = f.evaluate_model(train_abs_log, test_abs_log, net, im, fm, True)

    f.plot_petri_net(net, im, fm)
    f.plot_bpmn(net, im, fm)

    # Convert DataFrame to pm4py event log
    abstract_log = pm4py.convert_to_event_log(abstract_log, case_id_key="case:concept:name")

    # Export event log to XES format
    if params.get('export_net'):
        output_dir = "abstract"+clustering_tech+"_"+encoded_log
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        xes_exporter.apply(abstract_log, "{}/AbstractLog.xes".format(output_dir))
        pm4py.write_pnml(net, im, fm, "{}/AbstractModel.pnml".format(output_dir))

    return ['Abstract', clustering_tech, encoding, fitness, prec, gen, simp, f_score]



def merge_model(session_log, encoded_log, cluster_map, params):
    encoding = params.get('encoding')
    clustering_tech = params.get('clustering')
    activity_log, abstract_log = f.abstract_log_preprocessing(session_log, encoded_log, cluster_map)

    abstract_log = pm4py.convert_to_dataframe(abstract_log)

    # Step 1: Train inductive miner on high-level activities (clusters)
    net, im, fm = f.train_inductive_miner(abstract_log)
    # Step 2: Train inductive miners for each cluster
    clusters = [transition.label for transition in net.transitions if transition.label != None]
    miners = f.train_inductive_miners_for_clusters(activity_log, clusters)
    # Step 3: Replace individual miners with abstract high-level activity in the abstract model
    abstract_net = f.replace_transitions_with_miners(net, miners)
    print("Merge Model:")
    print("============\n")
    fitness, prec, gen, simp, f_score = f.evaluate_model(abstract_log, abstract_log, abstract_net, im, fm)

    f.plot_petri_net(abstract_net, im, fm)
    f.plot_bpmn(net, im, fm)
    print("*********************************")

    if params.get('export_net'):
        output_dir = "merge"+clustering_tech+"_"+encoded_log
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        xes_exporter.apply(activity_log, "{}/ActivityLog.xes".format(output_dir))
        pm4py.write_pnml(abstract_net, im, fm, "{}/MergeModel.pnml".format(output_dir))


    return ['Merge', clustering_tech, encoding, fitness, prec, gen, simp, f_score]