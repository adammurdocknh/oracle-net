import numpy
from numpy.lib.shape_base import column_stack

sheet = './sheets/LCS-2020-Summer_Player-Stats_OraclesElixir.csv'
# out_name = './sheets/lcs_2020_summer_player_stats_oracleselixir_scaled.csv'

out_name = './sheets/lcs_2020_summer_player_stats_oracleselixir_scaled.npy'

my_data = numpy.genfromtxt(sheet, delimiter=',')

num_rows, num_cols = my_data.shape

positions = my_data[:,0]
games_played = my_data[:,1]
win_rate = my_data[:,2]
column_to_be_added = numpy.column_stack((positions, games_played))
column_to_be_added = numpy.column_stack((column_to_be_added,win_rate))

# print(column_to_be_added)

numpy.delete(my_data,[0,2],1)
# print(positions)

def scale(my_data, x_min, x_max):
    nom = my_data-my_data.min(axis=0) * (x_max-x_min)
    denom = my_data.max(axis=0) - my_data.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom

my_data_scaled = scale(my_data,0,1)

my_data_scaled = column_stack((my_data_scaled,column_to_be_added))


# numpy.savetxt(out_name, my_data_scaled,delimiter=',')
numpy.save(out_name,my_data_scaled)