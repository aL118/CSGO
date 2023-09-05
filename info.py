import numpy as np

console = "D:\\lyzheng\\projects\\angela\\Counter-Strike_Behavioural_Cloning\\console_log.log"
data = open(console).read()
data_into_list = data.split('\n')
data_clean = []
for e in data_into_list:
    coords = e.split(" ")
    try:
        x = [float(coords[0]),float(coords[1]),float(coords[2])]
        data_clean.append(x)
    except:
        pass
