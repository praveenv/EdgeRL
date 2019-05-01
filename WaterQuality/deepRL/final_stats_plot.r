read_file <- read.csv("final_stats.csv",header=FALSE)

agent_stats = read_file$V1
truth_stats = read_file$V2
option_chosen_stats = read_file$V3


index = c()
for(i in 1:length(truth_stats))
{
	if(truth_stats[i] == 1 && option_chosen_stats[i] == 1)
		index = c(index,i)
}
print(index)