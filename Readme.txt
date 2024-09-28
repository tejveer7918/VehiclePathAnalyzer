Confidentiality Notice: This data package, including any data or video files is for the sole use of academic and research activities pertaining to Smart India Hackathon 2024. Any unauthorized use, copy, or distribution is prohibited and appropriate legal action will be taken.
© ISRO, 2024

The data package consists of geodetic points (latitude, longitude and height) [in *.kml files and *.pos files] collected using a GNSS receiver while travelling on Bengaluru-Mysuru Highway. Along with the geodetic data, video stream recorded from the vehicle is also provided to help understand movement of vehicle on the main road or service road. Description of each dataset is provided below:

Data/Video for modelling and testing:
1. Dataset1: Vehicle moving on main road with no exits or entry on the main road.

2. Dataset2: Vehicle moving on the service road with no entry or exits from the main road.

3. Dataset3: Vehicle starts on a service road, enters the main road, travels for a distance and then exits main road.

4. Dataset4: Vehicle starts on the main road, detours on a service road and then again enters the main road.

5. Dataset5: Vehicle moving on the main road which is also a flyover. Service road is below the flyover.

Instructions:
1. Note that the starting point of the vehicle can be anywhere: main road or service road, as described above. The algorithm must distinguish whether the vehicle is on the main road or a service road based on the given coordinates and the movement of the vehicle and then calculate the distance travelled by the vehicle on the main road in each dataset. The algorithm must detect entry/exit of the vehicle on the main road properly else the distance travelled will be incorrect.

2. One can use ISRO's Bhuvan or third party mapping tools e.g. Openstreet maps, Google maps, etc. for map matching. No assistance will be provided w.r.t mapping tools or APIs.

3. Developer is free to use any AI/ML tools to accomplish the goal.

4. Additional datasets will be provide during the Hackathon event to test and evaluate the developed solution.
