import pandas

if __name__ == '__main__':

    logs = [{'features': 'raw', 'model': 'rf', 'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.8145}}},
            {'features': 'raw', 'model': 'gbt', 'output': {'train': {'accuracy': 0.606}, 'test': {'accuracy': 0.4965}}},
            {'features': 'raw', 'model': 'hist_gbt',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.83}}},
            {'features': 'raw', 'model': 'ext', 'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.797}}},
            {'features': 'raw', 'model': 'pca_nn',
             'output': {'train': {'accuracy': 0.825}, 'test': {'accuracy': 0.731}}},
            {'features': 'raw', 'model': 'multiboosting',
             'output': {'train': {'accuracy': 0.7945}, 'test': {'accuracy': 0.758}}},
            {'features': 'raw', 'model': 'proximity_forest',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.7595}}},
            {'features': 'raw', 'model': 'rotation_forest',
             'output': {'train': {'accuracy': 0.695}, 'test': {'accuracy': 0.679}}},
            {'features': 'daisy', 'model': 'rf', 'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.8275}}},
            {'features': 'daisy', 'model': 'gbt', 'output': {'train': {'accuracy': 0.61}, 'test': {'accuracy': 0.514}}},
            {'features': 'daisy', 'model': 'hist_gbt',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.8435}}},
            {'features': 'daisy', 'model': 'ext', 'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.8075}}},
            {'features': 'daisy', 'model': 'pca_nn',
             'output': {'train': {'accuracy': 0.843}, 'test': {'accuracy': 0.7575}}},
            {'features': 'daisy', 'model': 'multiboosting',
             'output': {'train': {'accuracy': 0.8165}, 'test': {'accuracy': 0.765}}},
            {'features': 'daisy', 'model': 'proximity_forest',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.78}}},
            {'features': 'daisy', 'model': 'rotation_forest',
             'output': {'train': {'accuracy': 0.7085}, 'test': {'accuracy': 0.675}}},
            {'features': 'hog', 'model': 'rf', 'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.786}}},
            {'features': 'hog', 'model': 'gbt', 'output': {'train': {'accuracy': 0.133}, 'test': {'accuracy': 0.1455}}},
            {'features': 'hog', 'model': 'hist_gbt',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.8025}}},
            {'features': 'hog', 'model': 'ext', 'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.7685}}},
            {'features': 'hog', 'model': 'pca_nn',
             'output': {'train': {'accuracy': 0.818}, 'test': {'accuracy': 0.731}}},
            {'features': 'hog', 'model': 'multiboosting',
             'output': {'train': {'accuracy': 0.7815}, 'test': {'accuracy': 0.736}}},
            {'features': 'hog', 'model': 'proximity_forest',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.785}}},
            {'features': 'hog', 'model': 'rotation_forest',
             'output': {'train': {'accuracy': 0.718}, 'test': {'accuracy': 0.7125}}},
            {'features': 'rocket', 'model': 'rf', 'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.756}}},
            {'features': 'rocket', 'model': 'gbt',
             'output': {'train': {'accuracy': 0.7165}, 'test': {'accuracy': 0.556}}},
            {'features': 'rocket', 'model': 'hist_gbt',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.7865}}},
            {'features': 'rocket', 'model': 'ext',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.6535}}},
            {'features': 'rocket', 'model': 'pca_nn',
             'output': {'train': {'accuracy': 0.7015}, 'test': {'accuracy': 0.529}}},
            {'features': 'rocket', 'model': 'multiboosting',
             'output': {'train': {'accuracy': 0.7645}, 'test': {'accuracy': 0.7025}}},
            {'features': 'rocket', 'model': 'proximity_forest',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.4885}}},
            {'features': 'rocket', 'model': 'rotation_forest',
             'output': {'train': {'accuracy': 0.6275}, 'test': {'accuracy': 0.5955}}},
            {'features': 'unsup_km', 'model': 'rf',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.7705}}},
            {'features': 'unsup_km', 'model': 'gbt',
             'output': {'train': {'accuracy': 0.654}, 'test': {'accuracy': 0.565}}},
            {'features': 'unsup_km', 'model': 'hist_gbt',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.805}}},
            {'features': 'unsup_km', 'model': 'ext',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.769}}},
            {'features': 'unsup_km', 'model': 'pca_nn',
             'output': {'train': {'accuracy': 0.7955}, 'test': {'accuracy': 0.6975}}},
            {'features': 'unsup_km', 'model': 'multiboosting',
             'output': {'train': {'accuracy': 0.7925}, 'test': {'accuracy': 0.74}}},
            {'features': 'unsup_km', 'model': 'proximity_forest',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.7155}}},
            {'features': 'unsup_km', 'model': 'rotation_forest',
             'output': {'train': {'accuracy': 0.6825}, 'test': {'accuracy': 0.639}}},
            {'features': 'local', 'model': 'rf', 'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.8075}}},
            {'features': 'local', 'model': 'gbt',
             'output': {'train': {'accuracy': 0.612}, 'test': {'accuracy': 0.5035}}},
            {'features': 'local', 'model': 'hist_gbt',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.855}}},
            {'features': 'local', 'model': 'ext', 'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.759}}},
            {'features': 'local', 'model': 'pca_nn',
             'output': {'train': {'accuracy': 0.819}, 'test': {'accuracy': 0.733}}},
            {'features': 'local', 'model': 'multiboosting',
             'output': {'train': {'accuracy': 0.8275}, 'test': {'accuracy': 0.7725}}},
            {'features': 'local', 'model': 'proximity_forest',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.763}}},
            {'features': 'local', 'model': 'rotation_forest',
             'output': {'train': {'accuracy': 0.693}, 'test': {'accuracy': 0.6615}}},
            {'features': 'haar', 'model': 'rf', 'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.808}}},
            {'features': 'haar', 'model': 'gbt', 'output': {'train': {'accuracy': 0.675}, 'test': {'accuracy': 0.565}}},
            {'features': 'haar', 'model': 'hist_gbt',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.8585}}},
            {'features': 'haar', 'model': 'ext', 'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.7885}}},
            {'features': 'haar', 'model': 'pca_nn',
             'output': {'train': {'accuracy': 0.823}, 'test': {'accuracy': 0.715}}},
            {'features': 'haar', 'model': 'multiboosting',
             'output': {'train': {'accuracy': 0.8265}, 'test': {'accuracy': 0.771}}},
            {'features': 'haar', 'model': 'proximity_forest',
             'output': {'train': {'accuracy': 1.0}, 'test': {'accuracy': 0.7265}}},
            {'features': 'haar', 'model': 'rotation_forest',
             'output': {'train': {'accuracy': 0.7195}, 'test': {'accuracy': 0.6805}}}]

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
    print(df[["features", "model", "test_accuracy"]].sort_values(by="test_accuracy", ascending=False).to_markdown(index=False))