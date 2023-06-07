import numpy as np
import torch


def AJPE(V_pred, V_trgt, frame_idx):
    V_pred = V_pred - V_pred[:, :, :, 11:12, :]   # remove global translation
    V_trgt = V_trgt - V_trgt[:, :, :, 11:12, :]
    scale = 1000
    err = np.arange(len(frame_idx), dtype=np.float_)
    for idx in range(len(frame_idx)):
        temp = torch.norm(V_trgt[:, :, frame_idx[idx]-1, :, :] - V_pred[:, :, frame_idx[idx]-1, :, :], dim=3)
        err[idx] = temp.mean(1).mean()
    return err * scale


def GJPE(V_pred, V_trgt, frame_idx):
    scale = 1000
    err = np.arange(len(frame_idx), dtype=np.float_)
    for idx in range(len(frame_idx)):
        temp = torch.norm(V_trgt[:, :, frame_idx[idx]-1, :, :] - V_pred[:, :, frame_idx[idx]-1, :, :], dim=3).mean(-1)
        err[idx] = temp.mean(1).mean()
    return err * scale



def RFDE(V_pred,V_trgt, frame_idx):
    scale = 1000
    err = np.arange(len(frame_idx), dtype=np.float_)
    for idx in range(len(frame_idx)):
        temp = torch.linalg.norm(V_trgt[:, :, frame_idx[idx]-1:frame_idx[idx], : 1, :] - V_pred[:, :, frame_idx[idx]-1:frame_idx[idx], : 1, :], dim=-1)
        err[idx] = temp.mean(1).mean()
    return err * scale



def print_errors(scene, gjpe, ajpe, root):
    print()
    print(f"=============  {scene} ============== ")
    print(
        "{0: <16} | {1:6d} | {2:6d} | {3:6d} | {4:6d} | {5:6d} | {6:6d} | {7:6d}".format("Lengths",80,160,320,400,560,720,1000))
    print("=== JPE Test Error ===")
    print(
        "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f} | {5:6.0f} | {6:6.0f} | {7:6.0f}".format("Our", gjpe[0], gjpe[1], gjpe[2], gjpe[3],
                                                                                 gjpe[4], gjpe[5], gjpe[6]))
    print("=== APE Test Error ===")
    print(
        "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f} | {5:6.0f} | {6:6.0f} | {7:6.0f}".format("Our", ajpe[0], ajpe[1], ajpe[2], ajpe[3],
                                                                                 ajpe[4], ajpe[5], ajpe[6]))
    print("=== Root Test Error ===")
    print(
        "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f} | {5:6.0f} | {6:6.0f} | {7:6.0f}".format("Our", root[0], root[1], root[2],
                                                                                 root[3], root[4], root[5], root[6]))





# the below codes are from https://github.com/eth-ait/motion-transformer/blob/master/metrics/distribution_metrics.py

def power_spectrum(seq):
    """
    # seq = seq[:, :, 0:-1:12, :]  # 5 fps for amass (in 60 fps)
    
    Args:
      seq: (batch_size, n_joints, seq_len, feature_size)
  
    Returns:
        (n_joints, seq_len, feature_size)
        
    """
    feature_size = seq.shape[-1]
    n_joints = seq.shape[1]

    seq_t = np.transpose(seq, [0, 2, 1, 3])
    dims_to_use = np.where((np.reshape(seq_t, [-1, n_joints, feature_size]).std(0) >= 1e-6).all(axis=-1))[0]
    seq_t = seq_t[:, :, dims_to_use]

    seq_t = np.reshape(seq_t, [seq_t.shape[0], seq_t.shape[1], 1, -1])
    seq = np.transpose(seq_t, [0, 2, 1, 3])
    
    seq_fft = np.fft.fft(seq, axis=2)
    seq_ps = np.abs(seq_fft)**2
    
    seq_ps_global = seq_ps.sum(axis=0) + 1e-8
    seq_ps_global /= seq_ps_global.sum(axis=1, keepdims=True)

    seq_ps_global = seq_ps_global.reshape(seq.shape[2], -1, feature_size).transpose(1,0,2)
    return seq_ps_global


def ps_entropy(seq_ps):
    """
    
    Args:
        seq_ps: (n_joints, seq_len, feature_size)

    Returns:
    """
    return -np.sum(seq_ps * np.log(seq_ps), axis=1)


def ps_kld(seq_ps_from, seq_ps_to):
    """ Calculates KL(seq_ps_from, seq_ps_to).
    Args:
        seq_ps_from:
        seq_ps_to:

    Returns:
    """
    return np.sum(seq_ps_from * np.log(seq_ps_from / seq_ps_to), axis=1)

def long_term_metric(all_gt_test, all_pred, all_gt_target):

    # all_gt_test = np.zeros(0)
    
    train_data = np.load('./data/MI-Motion/data_train.npy', allow_pickle=True)
    bs, n, t, j, _ = train_data.shape
    all_gt_train = train_data.reshape(bs*n, t, j, -1).transpose(0,2,1,3)

    # for i in range(5):
    #     data =  np.load('./data/MI-Motion/data_test_S'+str(i)+'.npy', allow_pickle=True)
    #     bs, n, t, j, d = data.shape
    #     all_gt_test = np.vstack(data.reshape(bs*n, t, j, -1)) 
    # all_gt_test = all_gt_test.reshape(-1, t, j, d).transpose(0,2,1,3)
    
    pred_len = 25
    eval_seq_len = 5
    # print(f"all_gt_target:{all_gt_target[:, :, 0:pred_len:eval_seq_len, :].shape}")
    ps_gt_target = power_spectrum(all_gt_target[:, :, 0:pred_len:eval_seq_len, :])
    ps_gt_test = power_spectrum(all_gt_test[:, :, 0:pred_len:eval_seq_len, :])
    ps_gt_train = power_spectrum(all_gt_train[:, :, 0:pred_len:eval_seq_len, :])

    
    results = dict()
    results["entropy_gt_test"] = list()
    results["entropy_prediction"] = list()
    results["kld_prediction_target"] = list()
    results["kld_target_prediction"] = list()
    results["kld_test_prediction"] = list()
    results["kld_prediction_test"] = list()
    results["kld_train_prediction"] = list()
    results["kld_prediction_train"] = list()

    ent_gt_test = ps_entropy(ps_gt_test)
    results["entropy_gt_test"].append(ent_gt_test.mean())

    kld_train_test = ps_kld(ps_gt_train, ps_gt_test)
    kld_test_train = ps_kld(ps_gt_test, ps_gt_train)
    kld_test_target = ps_kld(ps_gt_test, ps_gt_target)
    kld_target_test = ps_kld(ps_gt_target, ps_gt_test)
    results["kld_train_test"] = kld_train_test.mean()
    results["kld_test_train"] = kld_test_train.mean()
    results["kld_test_target"] = kld_test_target.mean()
    results["kld_target_test"] = kld_target_test.mean()

    for sec, frame in enumerate(range(0, pred_len - eval_seq_len + 1, eval_seq_len)):
        ps_pred = power_spectrum(all_pred[:, :,frame:frame + eval_seq_len])

        ent_pred = ps_entropy(ps_pred)
        results["entropy_prediction"].append(ent_pred.mean())   

        kld_pred_target = ps_kld(ps_pred, ps_gt_target)
        results["kld_prediction_target"].append(kld_pred_target.mean())

        kld_target_pred = ps_kld(ps_gt_target, ps_pred)
        results["kld_target_prediction"].append(kld_target_pred.mean())

        kld_pred_test = ps_kld(ps_pred, ps_gt_test)
        results["kld_prediction_test"].append(kld_pred_test.mean())
        
        kld_test_pred = ps_kld(ps_gt_test, ps_pred)
        results["kld_test_prediction"].append(kld_test_pred.mean())

        kld_pred_train = ps_kld(ps_pred, ps_gt_train)
        results["kld_prediction_train"].append(kld_pred_train.mean())
        
        kld_train_pred = ps_kld(ps_gt_train, ps_pred)
        results["kld_train_prediction"].append(kld_train_pred.mean())  
   

    n_entries = len(results["entropy_prediction"])
    print("Test PS Entropy: {}".format(results["entropy_gt_test"]))  
    for sec in range(n_entries):
        i = str(sec + 1)
        print("Prediction Entropy-{}: {}".format(i, results["entropy_prediction"][sec]))


        # print("[{}] Prediction PS KLD: {}".format(sec + 1, (results["kld_prediction_target"][sec] + results["kld_target_prediction"][sec])/2))

    print("Train-Test PS KLD: {}".format((results["kld_test_train"] + results["kld_train_test"])/2))
    print("Test-Target PS KLD: {}".format((results["kld_test_target"] + results["kld_target_test"]) / 2))  # for complex crowd, because it does not exists in Train data
    for sec in range(n_entries):
        i = str(sec + 1)   
        print("Prediction KLD-{}: {}".format(i, (results["kld_test_prediction"][sec] + results["kld_prediction_test"][sec])/2))