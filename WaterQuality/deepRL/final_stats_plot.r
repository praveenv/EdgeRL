read_file <- read.csv("final_quality_stats_for_trace_simulation.csv",header=FALSE)

read_indices <- read.csv("../Data/non_ML_training_indices.csv")
indices = read_indices[,1]

read_time <- read.csv("../Data/HydroVu_1.csv",skip=7,header = TRUE, stringsAsFactors = FALSE)
sensor_date <- read_time[,1]
sensor_date <- strptime(sensor_date,format="%F %T")
# sensor_date <- strftime(sensor_date,format="%F %H")
sensor_date <- sensor_date[indices]
sensor_date <- sensor_date[-1]

agent_stats = read_file$V1
truth_stats = read_file$V2
option_chosen_stats = read_file$V3
fine_grained_stats = read_file$V4
coarse_grained_stats = read_file$V5

# index = c()

# event_count = 0
# correct_fine_count = 0
# incorrect_fine_count = 0
# correct_coarse_count = 0
# incorrect_coarse_count = 0

# for(i in 1:length(truth_stats))
# {
# 	if(truth_stats[i] == 1 && option_chosen_stats[i] == 1)
# 		index = c(index,i)

# 	if(truth_stats[i]==1)
# 	{
# 		event_count = event_count + 1
# 		if(option_chosen_stats[i]==1)
# 			correct_fine_count = correct_fine_count + 1
# 		else if(option_chosen_stats[i]==0)
# 			incorrect_fine_count = incorrect_fine_count + 1

# 	}
# 	if(truth_stats[i]==0)
# 	{
# 		if(option_chosen_stats[i]==0)
# 			correct_coarse_count = correct_coarse_count + 1
# 		else if(option_chosen_stats[i]==1)
# 			incorrect_coarse_count = incorrect_coarse_count + 1
# 	}
# }
# # print(index)
# # print(event_count)
# # print(correct_fine_count)
# # print(incorrect_fine_count)
# # print(correct_coarse_count)
# # print(incorrect_coarse_count)
sensor_date = sensor_date[1:length(truth_stats)]
df = data.frame(sensor_date,truth_stats,option_chosen_stats,fine_grained_stats,coarse_grained_stats)

write.csv(df,"quality_agent_truth_subplot.csv",row.names=FALSE)