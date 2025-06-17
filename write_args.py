import argparse

def write_args_file(file_path, no_of_uavs, no_of_ugvs, drl_model):
    """
    Writes a text file with specified arguments.

    Args:
        file_path (str): Path to the file to write.
        no_of_uavs (int): Number of UAVs.
        no_of_ugvs (int): Number of UGVs.
        drl_model (str): DRL model name.
    """
    content = f"""No_of_UAVs={no_of_uavs}
No_of_UGVs={no_of_ugvs}
DRL_model={drl_model}
"""
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        print(f"Arguments file written to {file_path}")
    except Exception as e:
        print(f"Error writing file: {e}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Write arguments to a text file.")
    parser.add_argument('--file_path', type=str, default="args.txt", help="Path to the output file")
    parser.add_argument('--no_of_uavs', type=int, required=True, help="Number of UAVs")
    parser.add_argument('--no_of_ugvs', type=int, required=True, help="Number of UGVs")
    parser.add_argument('--drl_model', type=str, required=True, help="DRL model name")

    args = parser.parse_args()

    # Write the file using the provided arguments
    write_args_file(args.file_path, args.no_of_uavs, args.no_of_ugvs, args.drl_model)