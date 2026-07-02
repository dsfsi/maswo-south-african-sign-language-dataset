import gzip
import pickle

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
    return loaded_object


def print_first_five(filename):
    # Load the file
    data = load_dataset_file(filename)

    # Print object type
    print("Type:", type(data))

    print("\nFirst 5 items:\n" + "-"*40)

    # Handle list, tuple, or any sequence
    if isinstance(data, (list, tuple)):
        for i, item in enumerate(data[:5]):
            print(f"[{i}] {item}")

    # Handle dict
    elif isinstance(data, dict):
        keys = list(data.keys())[:5]
        for i, k in enumerate(keys):
            print(f"[{i}] Key: {k} -> Value: {data[k]}")

    # If it's something else (custom class)
    else:
        try:
            # Try slicing
            for i, item in enumerate(data[:5]):
                print(f"[{i}] {item}")
        except:
            print("Object is not sliceable. Showing raw object:")
            print(data)


# ---------- RUN ----------
if __name__ == "__main__":
    filename = "./phoenix14t.pami0.test"   # change to your file path
    print_first_five(filename)
