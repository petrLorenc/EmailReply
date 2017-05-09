import re

import argparse


def load_file_to_process(path_to_file):
    with open(path_to_file, mode="r", encoding="ISO-8859-1") as file:
        lines = file.readlines()

    #nltk.download()

    pattern = r"^([A-Z]*):[a-z]* (.*)"

    input_x = []
    input_y = []
    for line in lines:
        matchObj = re.match(pattern, line.strip())
        yield matchObj.group(2), matchObj.group(1)

def main():
  parser = argparse.ArgumentParser(description='Description of your program')
  parser.add_argument('-i','--iFile', help='Input file', required=True)
  parser.add_argument('-oR','--oRawFile', help='Output File', required=True)
  parser.add_argument('-oT','--oTagFile', help='Output File', required=True)
  args = vars(parser.parse_args())
  inFile = args["iFile"]
  outFile = args["oRawFile"]
  outFileTag = args["oTagFile"]
  #procesed = load_file_to_process(inFile, args["stopwordFile"])

  with open(outFile, mode="w") as file:
    with open(outFileTag, mode="w") as fileTag:
        for line, tag in load_file_to_process(inFile):
          file.write(str(line) + "\n")
          fileTag.write(str(tag) + "\n")

if __name__ == "__main__":
    main()