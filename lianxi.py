logs = [
    {"api": "/login", "status": 200},
    {"api": "/login", "status": 500},
    {"api": "/user", "status": 200},
    {"api": "/user", "status": 200},
    {"api": "/user", "status": 404},
    {"api": "/order", "status": 500},
    {"api": "/order", "status": 500},
]
def lowest_success_api(logs):
    apps:dict[dict]={}
    apps2={}
    
    for log in logs:
        
        if log["api"] not in apps:
            
        
            apps[log["api"]]={"total":0,
                              "success":0}
        apps[log["api"]]["total"]+=1 
        if log["status"]==200:
            apps[log["api"]]["success"]+=1
    for key in apps.keys():
        apps2[key]=0
        

    
        
    return apps2
print(lowest_success_api(logs))
