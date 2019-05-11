read_file <- read.csv("../video_agent_truth_subplot.csv",header = TRUE,stringsAsFactors = FALSE)

timestamps <- read_file[,1]
option_chosen <- read_file[,4]

timestamps <- strptime(timestamps,format="%F %H:%M:%S")
timestamps <- strftime(timestamps,format="%a %H")

truth <- read_file[,2]

mat <- matrix(0,24,7)


current_row = 1
current_column = 1
unique_timestamps = unique(timestamps)
for(i in 1:length(unique_timestamps))
{
	current_timestamp = unique_timestamps[i]
	indices = which(timestamps == current_timestamp)
	values = option_chosen[indices]
	max_value = names(which.max(table(values))) 
	mat[current_row,current_column] = as.numeric(max_value)
	current_row = current_row + 1

	if(i %% 24 == 0)
	{
		current_column = current_column + 1
		current_row = 1
	}
}

print(mat)
write.csv(mat,"colormap_data.csv",row.names=FALSE)


current_row = 1
current_column = 1
mat <- matrix(0,24,7)

for(i in 1:length(unique_timestamps))
{
	current_timestamp = unique_timestamps[i]
	indices = which(timestamps==current_timestamp)
	values = truth[indices]
	max_value = max(values)
	mat[current_row,current_column] = as.numeric(max_value)
	current_row = current_row + 1

	if(i %% 24 == 0)
	{
		current_column = current_column + 1
		current_row = 1
	}
}

print(mat)
write.csv(mat,"colormap_truth.csv",row.names=FALSE)