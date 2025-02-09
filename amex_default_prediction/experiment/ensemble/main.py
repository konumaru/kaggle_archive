import numpy as np
import pandas as pd


def amex_metric_mod(y_true, y_pred):

    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:, 0] == 0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:, 0]) / np.sum(labels[:, 0])

    gini = [0, 0]
    for i in [1, 0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:, 0] == 0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] * weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1] / gini[0] + top_four)


def main():
    target = pd.read_parquet("./output/train_fe.parquet")["target"]

    ensemble_exp = [
        "exp_00109",  # xgb add te feature
        "exp_00200",  # lgbm
        "exp_00300",  # tabnet
    ]

    oofs = []
    for exp_name in ensemble_exp:
        oof = pd.read_csv(f"../{exp_name}/output/oof.csv")
        oofs.append(oof)

    oof = pd.concat(oofs, axis=1)
    oof_avg = oof.mean(axis=1)

    score = amex_metric_mod(target, oof_avg)
    print(score)

    submissions = []
    for exp_name in ensemble_exp:
        submission = pd.read_csv(f"../{exp_name}/output/submission.csv")
        submissions.append(submission)

    submission = pd.concat(submissions, axis=1)

    customer_id = pd.read_parquet("./output/test_fe.parquet")["customer_ID"]
    submission = pd.DataFrame(
        {
            "customer_ID": customer_id,
            "prediction": submission.mean(axis=1).to_numpy(),
        }
    )
    submission.to_csv("./output/submission.csv", index=False)


if __name__ == "__main__":
    main()
