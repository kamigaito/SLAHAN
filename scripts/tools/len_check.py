import argparse
import h5py

def options():
    parser = argparse.ArgumentParser(description='Google sentemce compression cleaner')
    parser.add_argument("-i", "--input-text", dest="input_text", type=str, help="path to the input file")
    parser.add_argument("-e", "--elmo-file", dest="elmo_file", type=str, help="path to an elmo style feature file")
    args = parser.parse_args()
    return args

def main():
    args = options()
    h5py_file = h5py.File(args.elmo_file, 'r')
    with open(args.input_text,"r") as f_in:
        sid = 0
        for line in f_in:
            embedding = h5py_file.get(str(sid))
            #assert(len(embedding[0]) == len(line.split(" ")))
            if  len(embedding[0]) != len(line.split(" ")):
                print(str(len(embedding[0])) + ", " + str(len(line.split(" "))))
                print(sid)
            sid += 1

if __name__ == "__main__":
    main()
