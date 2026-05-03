import tensorflow as tf
import numpy as np
from xgboost import XGBClassifier


def split_and_aggregate(image, aggregation_func, n_chunks=4):
    vert_splits = np.split(image, n_chunks, axis=0)
    splits = [np.split(v, n_chunks, axis=1) for v in vert_splits]
    aggregated = np.zeros((n_chunks, n_chunks), dtype=np.float32)

    for i in range(n_chunks):
        for j in range(n_chunks):
            aggregated[i, j] = aggregation_func(splits[i][j])
    return aggregated


def get_xgb_model():
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=10,
    )
    return model


def build_aggregated_dataset(x, aggregation_func, n_chunks=4):
    aggregated_images = []
    for img in x:
        aggregated_images.append(
            split_and_aggregate(img, aggregation_func, n_chunks).reshape(-1)
        )
    return np.array(aggregated_images)


def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    results = {}
    for func in [np.mean, np.median, np.max, np.min]:
        print(f"Running {func.__name__} for aggregation experiment...")
        x_train_aggregated = build_aggregated_dataset(x_train, func, n_chunks=4)
        x_test_aggregated = build_aggregated_dataset(x_test, func, n_chunks=4)
        model = get_xgb_model()
        model.fit(x_train_aggregated, y_train)
        score = model.score(x_test_aggregated, y_test)
        results[func.__name__] = score

    print("\nFinal Results:")
    for func_name, score in results.items():
        print(f"{func_name}: {score:.4f}")


if __name__ == "__main__":
    main()
