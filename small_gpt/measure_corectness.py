def measure_correctness(output: str):
    correct = 0
    
    arr = output.split("\n")
    
    print("arr: ", arr)
    
    for i in arr:    
        try:
            eval(i)
            correct += 1
        except:
            break
        
    print("correct: ", correct)