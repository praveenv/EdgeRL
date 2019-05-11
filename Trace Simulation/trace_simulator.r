agent_app_mapping <- function(index)
{
	if(index < 3)
		return(1)
	else
		return(2)
}



# EDGE SERVER GLOBAL STUFF

cpu_available <- 200
bandwidth_available <- 200
energy_available <- 200

# Application action plan cost matrices
# rows are the applications and the columns are their action plans (3 columns - coarse, medium, fine) -1 if no medium
# first row is water and second row is video 
ap_cpu <- matrix(c(5,5,-1,30,15,60),nrow=2,ncol=3)
ap_bandwidth <- matrix(c(5,5,-1,20,10,35),nrow=2,ncol=3)
ap_energy <- matrix(c(10,10,-1,30,15,60),nrow=2,ncol=3)

data <- read.csv("./Data/combined_data.csv",header=TRUE,stringsAsFactors=FALSE)
timestamps <- data[,1]
data <- as.matrix(data)

priority_score = c(2,5,1,3,4,6,7) # first 2 are water and next 5 are camera

priority_order = c(7,6,2,5,4,1,3) # corresponding order of above priority score


agent_columns = c(3,7,12,18,24,30,36)
truth_columns = c(2,6,10,16,22,28,34)
fine_columns = c(4,8,14,20,26,32,38)
coarse_columns = c(5,9,13,19,25,31,37)
medium_columns = c(1,1,15,21,27,33,39)

tp_count = rep(0,length(agent_columns))
tn_count = rep(0,length(agent_columns))
fp_count = rep(0,length(agent_columns))
fn_count = rep(0,length(agent_columns))

distance = rep(0,length(agent_columns))
total_people = rep(0,length(agent_columns))

cpu_cost_incurred = 0
bandwidth_cost_incurred = 0
energy_cost_incurred = 0

# print(ap_cpu)
# print(ap_bandwidth)
# print(ap_energy)
app_count = rep(0,2)

agent_option_per_timestep <- c()

number_of_request_denied = 0

for(i in 1:(length(timestamps)-1))
# for(i in 1:1)
{
	cpu_available <- 150
	bandwidth_available <- 150
	energy_available <- 150
	
	agent_values = as.numeric(data[i,agent_columns])
	# print(agent_values)
	
	agent_option_current_timestep = rep(-1,length(agent_values))
	for(j in 1:length(agent_values))
	{
		current_agent = priority_order[j]
		if(agent_values[current_agent] == -1)
			next
		current_app = agent_app_mapping(current_agent)
		
		current_value = agent_values[current_agent]
		app_count[current_app] = app_count[current_app] + 1
		temp_cpu = cpu_available - ap_cpu[current_app,current_value+1]
		temp_bandwidth = bandwidth_available - ap_bandwidth[current_app,current_value+1]
		temp_energy = energy_available - ap_energy[current_app,current_value+1]
		if(temp_cpu < 0 || temp_bandwidth < 0 || temp_energy < 0)
		{
			agent_option_current_timestep[current_agent] = 0
			cpu_available = cpu_available - ap_cpu[current_app,1]
			bandwidth_available = bandwidth_available - ap_bandwidth[current_app,1]
			energy_available = energy_available - ap_energy[current_app,1]
			number_of_request_denied = number_of_request_denied + 1

			cpu_cost_incurred = cpu_cost_incurred + ap_cpu[current_app,1]
			bandwidth_cost_incurred = bandwidth_cost_incurred + ap_bandwidth[current_app,1]
			energy_cost_incurred = energy_cost_incurred + ap_energy[current_app,1]

			if(current_value == 0)
			{
				current_truth = as.numeric(data[i+1,truth_columns[current_agent]])
				current_prediction = as.numeric(data[i+1,coarse_columns[current_agent]])
				if(current_app == 1) # current application is water
				{
					current_truth = as.numeric(data[i,truth_columns[current_agent]])
					current_prediction = as.numeric(data[i,coarse_columns[current_agent]])
					if(current_truth == 0 && current_prediction == 0)
						tn_count[current_agent] = tn_count[current_agent] + 1
					if(current_truth == 1 && current_prediction == 1)
						tp_count[current_agent] = tp_count[current_agent] + 1
					if(current_truth == 1 && current_prediction == 0)
						fn_count[current_agent] = fn_count[current_agent] + 1
					if(current_truth == 0 && current_prediction == 1)
						fp_count[current_agent] = fp_count[current_agent] + 1
				}
				else
				{
					distance[current_agent] = distance[current_agent] + abs(current_truth - current_prediction)	
					total_people[current_agent] = total_people[current_agent] + current_truth
				}
			}

			if(current_value == 1)
			{
				current_truth = as.numeric(data[i+1,truth_columns[current_agent]])
				current_prediction = as.numeric(data[i+1,medium_columns[current_agent]])
				if(current_app == 1) # current application is water
				{
					next
				}
				else
				{
					distance[current_agent] = distance[current_agent] + abs(current_truth - current_prediction)
					total_people[current_agent] = total_people[current_agent] + current_truth	
				}
			}

			if(current_value == 2)
			{
				current_truth = as.numeric(data[i+1,truth_columns[current_agent]])
				current_prediction = as.numeric(data[i+1,fine_columns[current_agent]])
				if(current_app == 1) # current application is water
				{
					current_truth = as.numeric(data[i,truth_columns[current_agent]])
					current_prediction = as.numeric(data[i,coarse_columns[current_agent]])
					if(current_truth == 0 && current_prediction == 0)
						tn_count[current_agent] = tn_count[current_agent] + 1
					if(current_truth == 1 && current_prediction == 1)
						tp_count[current_agent] = tp_count[current_agent] + 1
					if(current_truth == 1 && current_prediction == 0)
						fn_count[current_agent] = fn_count[current_agent] + 1
					if(current_truth == 0 && current_prediction == 1)
						fp_count[current_agent] = fp_count[current_agent] + 1


				}
				else
				{
					distance[current_agent] = distance[current_agent] + abs(current_truth - current_prediction)
					total_people[current_agent] = total_people[current_agent] + current_truth	
				}
			}
		}
		else
		{
			agent_option_current_timestep[current_agent] = current_value
			cpu_available = cpu_available - ap_cpu[current_app,current_value+1]
			bandwidth_available = bandwidth_available - ap_bandwidth[current_app,current_value+1]
			energy_available = energy_available - ap_energy[current_app,current_value+1]

			cpu_cost_incurred = cpu_cost_incurred + ap_cpu[current_app,current_value+1]
			bandwidth_cost_incurred = bandwidth_cost_incurred + ap_bandwidth[current_app,current_value+1]
			energy_cost_incurred = energy_cost_incurred + ap_energy[current_app,current_value+1]

			if(current_value == 0)
			{
				current_truth = as.numeric(data[i+1,truth_columns[current_agent]])
				current_prediction = as.numeric(data[i+1,coarse_columns[current_agent]])
				if(current_app == 1) # current application is water
				{
					current_truth = as.numeric(data[i,truth_columns[current_agent]])
					current_prediction = as.numeric(data[i,coarse_columns[current_agent]])
					if(current_truth == 0 && current_prediction == 0)
						tn_count[current_agent] = tn_count[current_agent] + 1
					if(current_truth == 1 && current_prediction == 1)
						tp_count[current_agent] = tp_count[current_agent] + 1
					if(current_truth == 1 && current_prediction == 0)
						fn_count[current_agent] = fn_count[current_agent] + 1
					if(current_truth == 0 && current_prediction == 1)
						fp_count[current_agent] = fp_count[current_agent] + 1
				}
				else
				{
					distance[current_agent] = distance[current_agent] + abs(current_truth - current_prediction)
					total_people[current_agent] = total_people[current_agent] + current_truth	
				}
			}

			if(current_value == 1)
			{
				current_truth = as.numeric(data[i+1,truth_columns[current_agent]])
				current_prediction = as.numeric(data[i+1,medium_columns[current_agent]])
				if(current_app == 1) # current application is water
				{
					next
				}
				else
				{
					distance[current_agent] = distance[current_agent] + abs(current_truth - current_prediction)	
					total_people[current_agent] = total_people[current_agent] + current_truth
				}
			}

			if(current_value == 2)
			{
				current_truth = as.numeric(data[i+1,truth_columns[current_agent]])
				current_prediction = as.numeric(data[i+1,fine_columns[current_agent]])
				if(current_app == 1) # current application is water
				{
					current_truth = as.numeric(data[i,truth_columns[current_agent]])
					current_prediction = as.numeric(data[i,coarse_columns[current_agent]])
					if(current_truth == 0 && current_prediction == 0)
						tn_count[current_agent] = tn_count[current_agent] + 1
					if(current_truth == 1 && current_prediction == 1)
						tp_count[current_agent] = tp_count[current_agent] + 1
					if(current_truth == 1 && current_prediction == 0)
						fn_count[current_agent] = fn_count[current_agent] + 1
					if(current_truth == 0 && current_prediction == 1)
						fp_count[current_agent] = fp_count[current_agent] + 1

				}
				else
				{

					distance[current_agent] = distance[current_agent] + abs(current_truth - current_prediction)
					total_people[current_agent] = total_people[current_agent] + current_truth	
				}
			}
		}
		# print("AVAILABLE STUFF")
		# print(cpu_available)
		# print(bandwidth_available)
		# print(energy_available)
	}
	agent_option_per_timestep = rbind(agent_option_per_timestep,agent_option_current_timestep)

}

print("STATS 1")
# print(agent_option_per_timestep)
print(number_of_request_denied)
print(cpu_cost_incurred)
print(bandwidth_cost_incurred)
print(energy_cost_incurred)
print(tp_count)
print(tn_count)
print(fp_count)
print(fn_count)
print(distance)
print(total_people)
print(app_count)
