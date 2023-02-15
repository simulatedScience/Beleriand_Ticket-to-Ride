import os
from typing import List

def split_language_file(filepath: str, languages: List[str], seperator: str = " ; ") -> None:
  """
  Split a TTR location file (.txt) into multiple files, one for each language.
  Each row of the original file should contain location names in the specified languages in the same order.
  The new files will be named after the original file, with the language code appended to the end of the name. They will contain the location names for the corresponding language.
  Empty lines will be ignored.

  Args:
      filepath (str): Path to the location file to split.
      languages (List[str]): List of language codes for the languages in the file (order matters)
      seperator (str, optional): Seperator between location names in the file. Defaults to " ; ".
  """
  # Get the file name without the extension
  filename = os.path.splitext(os.path.basename(filepath))[0]

  # Read the file
  with open(filepath, "r", encoding="utf-8") as f:
    lines = f.readlines()

  # Create a list of files, one for each language
  files = []
  for language in languages:
    new_filename = f"{filename}_{language}.txt"
    # create new files in the same directory as the original file
    new_filepath = os.path.join(os.path.dirname(filepath), new_filename)
    files.append(open(new_filepath, "w", encoding="utf-8"))

  # Write the location names to the corresponding files
  for line in lines:
    # Ignore empty lines
    if line.strip() == "":
      continue
    # Split the line into location names
    locations = line.split(seperator)
    # Write the location names to the corresponding files
    for i in range(len(languages)):
      files[i].write(locations[i])
      if not locations[i].endswith("\n"):
        files[i].write("\n")

  # Close the files
  for file in files:
    file.close()

if __name__ == "__main__":
  # filename = "dq_9_locations_bilingual.txt"
  # filepath = os.path.join(os.getcwd(), "dq_9_ttr", filename)
  filename = "ttr_europe_cities.txt"
  filepath = os.path.join(os.getcwd(), "europe_ttr", filename)
  languages = ["de", "en"]
  split_language_file(filepath, languages)