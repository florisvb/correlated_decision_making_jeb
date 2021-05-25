import fly_plot_lib.flymath as flymath
import pandas
import pickle
import numpy as np

def load_pickle():
    f = open('time_when_food_found_dark.pickle')
    time_when_food_found = pickle.load(f)
    f.close()
    return time_when_food_found
    
def get_frame_where_food_found(near_food):
    chunks, breaks = flymath.get_continuous_chunks(near_food)
    for chunk in chunks:
        if len(chunk) > 12: # > 0.2 seconds:
            return chunk[0]
    return None

def get_time_to_find_food(fly_number, arena, flies):
    food_x = arena.query('fly=='+str(fly_number))['food_center_x'].values[0]
    food_y = arena.query('fly=='+str(fly_number))['food_center_y'].values[0]

    fly = flies.query('fly=='+str(fly_number))
    fly['distance_to_food'] = np.sqrt((fly.position_x-food_x)**2 + (fly.position_y-food_y)**2)
    near_food = np.where(fly.distance_to_food<1.5)[0]
    
    if len(near_food) > 0:
        frame_where_food_found = get_frame_where_food_found(near_food)
        if frame_where_food_found is not None:
            time_when_food_found = fly.time.iloc[frame_where_food_found]
            return time_when_food_found
    return None

if __name__ == '__main__':

    arena = pandas.read_csv('dataset_03_largeArena_dark_arena.csv')
    flies = pandas.read_csv('dataset_03_largeArena_dark_flies.csv')

    times_when_food_found = []

    for fly_number in flies.fly.unique():
        t = get_time_to_find_food(fly_number, arena, flies)
        if t is not None:
            times_when_food_found.append(t)

    f = open('time_when_food_found_dark.pickle', 'w')
    pickle.dump(times_when_food_found, f)
    f.close()