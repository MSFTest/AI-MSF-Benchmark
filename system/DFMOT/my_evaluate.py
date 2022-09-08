from kitti_devkit.evaluate_tracking import evaluate

if __name__ == '__main__':
    step = "train"
    part = "all"
    result_path = "results"

    X = evaluate(
        step, result_path, part=part)
    print(X)
    MOTA, MOTP, recall, prec, F1, fp, fn, id_switches = X
