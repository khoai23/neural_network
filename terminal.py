import sys

def createTerminal(data, commandDict, helperDict=None, initialDisplayString=None, externalTerminateCondition=None):
	if(initialDisplayString):
		print(initialDisplayString)
	
	if(externalTerminateCondition is None):
		externalTerminateCondition = lambda *args: True
	
	exitAll = lambda *args: sys.exit()
	# override q/quit options with the exit
	commandDict["q"] = exitAll
	commandDict["quit"] = exitAll

	while(externalTerminateCondition(data)):
		commandKey = input("Please insert your command here: ").strip()
		if(commandKey in commandDict):
			command = commandDict[commandKey]
			if(helperDict and commandKey in helperDict):
				print("Helper: {}".format(helperDict[commandKey]))
			assert callable(command)
			newData = command(data)
			if(newData is not None):
				data = newData
		else:
			print("Invalid command. All commands: ", ", ".join(commandDict.keys()))
	
	print("Terminate condition reached. Exiting.")
	sys.exit()
