library("ggplot2")
library("gridExtra")
library("cowplot")

quality_agent_truth_subplot <- function(filepath)
{
	read_file <- read.csv(filepath,header=FALSE)
	timestamps <- data.frame(read_file[,1])
	colnames(timestamps) <- c("timestamps")
	truth <- data.frame(read_file[,2])
	colnames(truth) <- c("truth")
	agent <- data.frame(read_file[,3])
	colnames(agent) <- c("agent")
	timestamps = data.frame(seq(1, 529, 1))

	colnames(timestamps) <- c("timestamps")
	df = cbind(timestamps,truth,agent)


	plot.truth <- ggplot() + geom_line(aes(y = truth, x = timestamps),
                           data = df, stat="identity") + scale_x_continuous(name = "Timestamps",breaks=c(0,529),labels=c("08/14 00:00","08/22 00:00")) + scale_y_continuous(name = "Ground Truth",breaks=c(0.00,1.00),labels=c("No Contamination","Contamination")) + theme_classic() + theme(axis.text.x= element_text(face="bold")) + theme(axis.text.y= element_text(face="bold"))


	plot.agent <- ggplot() + geom_line(aes(y = agent, x = timestamps),
                           data = df, stat="identity") + scale_x_continuous(name = "Timestamps",breaks=c(0,529),labels=c("08/14 00:00","08/22 00:00")) + scale_y_continuous(name = "Agent Option Chosen",breaks=c(0.00,1.00),labels=c("No Contamination","Contamination")) + theme_classic() + theme(axis.text.x= element_text(face="bold")) + theme(axis.text.y= element_text(face="bold"))

	# ggsave(p1,'./quality_agent_truth_subplot_1.png', dpi=100)

	final_plot <- grid.arrange(plot.truth,plot.agent,ncol=1)
	ggsave("quality_agent_truth_subplot.eps",final_plot,dpi=100,device="eps")
}


# labels=c("0" = "08/14 00:00","529" = "08/22 00:00")
# + scale_y_continuous(name = "Ground Truth",breaks=c(0.00,1.00),labels=c("0.00"="No Contamination","1.00"="Contamination"))

# MAIN SECTION

quality_agent_truth_subplot("./quality_agent_truth_subplot.csv")