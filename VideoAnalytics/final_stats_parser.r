read_days <- read.csv("./Data/combined_days.csv",header = TRUE,stringsAsFactors=FALSE)
read_stats <- read.csv("final_stats_all_days_new_with_medium.csv",header = FALSE)

print(head(read_days))
print(head(read_stats))

read_days = read_days[-1,]

timestamps <- read_days[,1]
truth <- read_stats[,2]
agent <- read_stats[,1]
option_chosen <- read_stats[,3]
motion_sensor <- read_stats[,4]
yolo_output <- read_stats[,5]
opencv_output <- read_stats[,6]

df <- data.frame(timestamps,truth,agent,option_chosen,motion_sensor,yolo_output,opencv_output)
colnames(df) <- c("timestamps","truth","agent","option_chosen","motion_sensor","yolo_output","opencv_output")

write.csv(df,"./plot_data/video_agent_truth_subplot.csv",row.names=FALSE)
