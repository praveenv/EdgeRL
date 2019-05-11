water_map <- function(timestamps,water_data)
{
	mat <- matrix(-1,length(timestamps),4)
	for(i in 1:length(water_data[,1]))
	{
		index = which(timestamps == water_data[i,1])
		replacement = as.matrix(water_data[i,2:5])
		mat[index,] <- replacement
	}
	return(mat)
}


video_map <- function(timestamps,video_data)
{
	mat <- matrix(-1,length(timestamps),6)
	for(i in 1:length(video_data[,1]))
	{
		index = which(timestamps == video_data[i,1])
		replacement = as.matrix(video_data[i,2:7])
		mat[index,] <- replacement
	}
	return(mat)
}

read_video_data <- read.csv("./Data/video_agent_truth_subplot.csv")
read_quality_data <- read.csv("./Data/quality_agent_truth_subplot.csv")

video_timestamp <- strptime(read_video_data[,1],format="%F %H:%M:%S")
video_timestamp <- strftime(video_timestamp,format="%H:%M:%S")
read_video_data[,1] <- video_timestamp

quality_timestamp <- strptime(read_quality_data[,1],format="%F %H:%M:%S")
quality_timestamp <- strftime(quality_timestamp,format="%H:%M:%S")
read_quality_data[,1] <- quality_timestamp

camera_1 <- read_video_data[1:1439,]
camera_2 <- read_video_data[1440:2879,]
camera_3 <- read_video_data[2880:4319,]
camera_4 <- read_video_data[4320:5759,]
camera_5 <- read_video_data[5760:7199,]

water_1 <- read_quality_data[4012:4088,]
water_2 <- read_quality_data[4089:4169,]

water_1[which(water_1[,3]==1),3] = 2
water_2[which(water_2[,3]==1),3] = 2


starting_time <- "00:00:00"
starting_time <- strptime(starting_time,format="%H:%M:%S")
starting_time = seq(starting_time,starting_time + (24*60*60) - 1, 60)
timestamps = strftime(starting_time,format="%H:%M:%S")

water_1 <- water_map(timestamps,water_1)
colnames(water_1) <- c("water1_truth","water1_option","water1_fine","water1_coarse")
water_2 <- water_map(timestamps,water_2)
colnames(water_2) <- c("water2_truth","water2_option","water2_fine","water2_coarse")

camera_1 <- video_map(timestamps,camera_1)
colnames(camera_1) <- c("camera1_truth","camera1_agent","camera1_option","camera1_motion","camera1_yolo","camera1_opencv")
camera_2 <- video_map(timestamps,camera_2)
colnames(camera_2) <- c("camera2_truth","camera2_agent","camera2_option","camera2_motion","camera2_yolo","camera2_opencv")
camera_3 <- video_map(timestamps,camera_3)
colnames(camera_3) <- c("camera3_truth","camera3_agent","camera3_option","camera3_motion","camera3_yolo","camera3_opencv")
camera_4 <- video_map(timestamps,camera_4)
colnames(camera_4) <- c("camera4_truth","camera4_agent","camera4_option","camera4_motion","camera4_yolo","camera4_opencv")
camera_5 <- video_map(timestamps,camera_5)
colnames(camera_5) <- c("camera5_truth","camera5_agent","camera5_option","camera5_motion","camera5_yolo","camera5_opencv")

final_df <- cbind(timestamps,water_1,water_2,camera_1,camera_2,camera_3,camera_4,camera_5)

write.csv(final_df,"./Data/combined_data.csv",row.names=FALSE)