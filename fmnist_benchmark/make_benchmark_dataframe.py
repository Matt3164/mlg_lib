import pandas

if __name__ == '__main__':

    logs = [{'features': 'raw', 'model': 'rf', 'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.8225}}},
            {'features': 'raw', 'model': 'gbt',
             'output': {'train': {'accuracy': 0.6395}, 'test': {'accuracy': 0.5335}}},
            {'features': 'raw', 'model': 'hist_gbt',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.83}}},
            {'features': 'raw', 'model': 'ext', 'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.7895}}},
            {'features': 'raw', 'model': 'pca_nn',
             'output': {'train': {'accuracy': 0.825}, 'test': {'accuracy': 0.731}}},
            {'features': 'daisy', 'model': 'rf', 'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.829}}},
            {'features': 'daisy', 'model': 'gbt',
             'output': {'train': {'accuracy': 0.793}, 'test': {'accuracy': 0.649}}},
            {'features': 'daisy', 'model': 'hist_gbt',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.8435}}},
            {'features': 'daisy', 'model': 'ext', 'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.811}}},
            {'features': 'daisy', 'model': 'pca_nn',
             'output': {'train': {'accuracy': 0.843}, 'test': {'accuracy': 0.7575}}},
            {'features': 'hog', 'model': 'rf', 'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.7875}}},
            {'features': 'hog', 'model': 'gbt', 'output': {'train': {'accuracy': 0.6775}, 'test': {'accuracy': 0.506}}},
            {'features': 'hog', 'model': 'hist_gbt',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.8025}}},
            {'features': 'hog', 'model': 'ext', 'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.772}}},
            {'features': 'hog', 'model': 'pca_nn',
             'output': {'train': {'accuracy': 0.818}, 'test': {'accuracy': 0.731}}},
            {'features': 'rocket', 'model': 'rf', 'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.7555}}},
            {'features': 'rocket', 'model': 'gbt',
             'output': {'train': {'accuracy': 0.7005}, 'test': {'accuracy': 0.5635}}},
            {'features': 'rocket', 'model': 'hist_gbt',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.7865}}},
            {'features': 'rocket', 'model': 'ext',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.6565}}},
            {'features': 'rocket', 'model': 'pca_nn',
             'output': {'train': {'accuracy': 0.7015}, 'test': {'accuracy': 0.529}}},
            {'features': 'unsup_km', 'model': 'rf',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.775}}},
            {'features': 'unsup_km', 'model': 'gbt',
             'output': {'train': {'accuracy': 0.725}, 'test': {'accuracy': 0.6005}}},
            {'features': 'unsup_km', 'model': 'hist_gbt',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.805}}},
            {'features': 'unsup_km', 'model': 'ext',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.7655}}},
            {'features': 'unsup_km', 'model': 'pca_nn',
             'output': {'train': {'accuracy': 0.7955}, 'test': {'accuracy': 0.6975}}}]

    rows = list()
    for res in logs:
        rows.append(
            {
                "test_accuracy": res["output"]["test"]["accuracy"],
                "train_accuracy": res["output"]["train"]["accuracy"],
                "model": res["model"],
                "features": res["features"],
            }
        )

    df = pandas.DataFrame(rows)
    print(df.sort_values(by="test_accuracy",ascending=False).to_markdown(index=False))
