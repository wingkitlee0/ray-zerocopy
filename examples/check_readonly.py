import numpy as np
import torch
import warnings


def check():
    # Create a read-only numpy array
    arr = np.zeros(10)
    arr.flags.writeable = False

    print(f"Array writable: {arr.flags.writeable}")

    print("Trying torch.as_tensor(arr)...")
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            t = torch.as_tensor(arr)
            print(
                f"Success! Tensor created. Share memory? {np.shares_memory(t.numpy(), arr)}"
            )
            if len(w) > 0:
                print(f"Warnings: {[str(x.message) for x in w]}")
    except Exception as e:
        print(f"Failed with error: {e}")


if __name__ == "__main__":
    check()
