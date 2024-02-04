import sys
import numpy as np
from src import GeneticAlgorithm
from src.utils import binary_ops
import numpy as np
import enum

# class OptimizationExceptionFactory(object):
    

class OptimizationException(Exception):
    def __init__(self,*args):
        super().__init__(args)
        self.__penalty = 1
    @property
    def penalty(self):
        return self.__penalty
    
   



class ProhibitedExceptions(OptimizationException):
    def __init__(self,message):
        super().__init__(message)
        self._OptimizationException__penalty = 10**10  
class NotRecommendedException(OptimizationException):
    def __init__(self,message):
        super().__init__(message)
        self._OptimizationException__penalty = 10**7
class OnlyOnLowResources(OptimizationException):
    def __init__(self,message):
        super().__init__(message)
        self._OptimizationException__penalty = 10**4




class LogisticException(ProhibitedExceptions):
    pass
class LogicError(ProhibitedExceptions):
    pass
class VehicleMaxVolumeExeeded(OnlyOnLowResources):
    pass 


class SameLocationException(NotRecommendedException):
    pass
class OriginNonExistentException(NotRecommendedException):
    pass
class DestinyNonExistentException(ProhibitedExceptions):
    pass
class VehicleStoppedException(NotRecommendedException):
    pass



class Depot(object):
    def __init__(self):
        self.__cargos = []
        self.__docks = None
    @property
    def docks(self):
        return self.__docks
    @docks.setter
    def docks(self,docks):
        # setter to make validations
        self.__docks = docks
    @property
    def cargos(self):
        return self.__cargos
    @cargos.setter
    def cargos(self,cargos):
        self.__cargos = cargos 
    
class Dock(object):
    def __init__(self,depot):
        self.__current_vehicle = None
        self.__depot = depot
    
    def getCargoForCurrentVehicle(self):
        self._Dock__depot.getCargo(self.__current_vehicle)
        
class Cargo(object):
    def __init__(self, cargo_data):
        '''
            Cargo(cargo_data)
            
            if cargo_data is an array: 
                #0 - origin_id,
                #1 - dest_id,
                #2 - cargo_id,
                #3 - volume(m3),
                #4 - max_stack,
                #5 - cubagem(kg/m3 - Densidade),
                #6 - peso(kg)

        '''
        if isinstance(cargo_data,np.ndarray) | isinstance(cargo_data,list):
            
            self.__origin    = cargo_data[0]
            self.__destiny   = cargo_data[1]
            self.__cargo_id  = cargo_data[2]
            self.__volume    = cargo_data[3]
            # self.__max_stack = cargo_data[4]
            # self.__cubagem   = cargo_data[5]
            # self.__peso      = cargo_data[6]
            
        elif(type(cargo_data) == dict):
            raise Exception("Not Implemented")

        self.__vehicle = None
        self.listener = lambda err: print(err)    
        self.__pickup = False
        self.__delivered = False
    
    @property
    def max_stack(self):
        return self.__max_stack

    @property
    def destiny(self):
        return self.__destiny
    @property
    def origin(self):
        return self.__origin
    
    @property
    def delivered(self):
        return self.__delivered
    @property
    def pickuped(self):
        return self.__pickup

    @property
    def volume(self):
        return self.__volume

    @property
    def code(self):
        return self.__cargo_id

    # @description.setter
    def description(self,name):
        self.__name = name
    
    @property
    def description(self):
        return self.__name
    
    @property
    def vehicle(self):
        return self.__vehicle 

    @vehicle.setter
    def vehicle(self,vehicle):
        if self.__origin not in vehicle.route:
            self.listener(OriginNonExistentException("The origin {}, of cargo {}, are not part of the vehicle {} route.".format(self.__origin,self.code,vehicle.code)))
            self.__pickup = False
        else: 
            self.__pickup = True
            pass        
        if (not self.__pickup) | (self.__destiny not in vehicle.route[1:]):
            self.listener(DestinyNonExistentException("The destiny {}, of cargo {}, are not part of the vehicle {} route.".format(self.__origin,self.code,vehicle.code)))
            self.__delivered = False
        else:
            self.__delivered = True
            pass

        if self.__vehicle is None:
            self.__vehicle = vehicle
        else:
            self.listener(LogisticException("Cargo is already registered in another vehicle."))
        
    def calculateMetrics(self,data):
        # time_to_delivery
        # distance_traveled
        pass
        


    def printMetrics(self):
        print(f"Cargo {self.code} on {self.vehicle.code if self.vehicle else 'None'}: {self.origin}({self.pickuped})->{self.destiny}({self.delivered})  ")           

class Vehicle(object):

    def __init__(self, vehicle_data,vehicle_name="General Vehicle"):
        self.__cargo = {}
        self.__id = None
        self.__route = None
        self.__location = None
        self.__volume = None
        self.__type_code = None
        self.__name = vehicle_name
        # self.name = vehicle_name
        self.listener = lambda err: print(err)
        self.__moviment_tuples = None
        if isinstance(vehicle_data,list):
            vehicle_data = np.array(vehicle_data)

        if type(vehicle_data) == np.ndarray:
            self.__id = vehicle_data[1]
            self.__location = vehicle_data[2]
        


    
        # statistics
        self.total_dist = 0
        self.total_time = 0
        self.stops = 0
        self.times = []
        self.dists = []
    
    @property
    def name(self):
        return self._Vehicle__name

    @property
    def max_volume(self):
        return self._Vehicle__volume
        
    @property
    def current_volume(self):
        return sum(self.list_volumes)
    
    @property
    def list_volumes(self):
        return np.array(list(map(lambda cargo: cargo.volume,self.iter_cargo)))
    
    @property
    def route(self):
        return self.__route
    
    @property
    def location(self):
        return self.__location

    @route.setter
    def route(self,route_movements):
        argsort = np.argsort(route_movements)
        argmin  = np.argmin(route_movements)
        minisstop = route_movements[argmin]<0

        self.__route = route_movements[:argmin] if minisstop else route_movements
        self.__route = np.concatenate(([self.__location],self.__route))

        
        for t in self.from_to_tuples:
            if t[0]-t[1] == 0:
                self.listener(SameLocationException(f"Vehicle {self.code} are going to the same location({t[0]},{t[1]})"))

    @property
    def from_to_tuples(self):
        if self.__moviment_tuples is None:
            self.__moviment_tuples = list(moviment_tuple(self.route))
        return self.__moviment_tuples

    @property
    def code(self):
        return self.__id
    
    @property
    def stopped(self):
        return len(self.from_to_tuples)==0
    
    @property
    def iter_cargo(self): 
        for key in self.__cargo:
            yield self.__cargo[key]
    
    def addCargo(self,cargo):
        if cargo.code in self.__cargo:
            # print('Cargo is already added to this vehicle.')
            pass
        self.__cargo[cargo.code]=cargo
        cargo.vehicle = self

    def calculate_metric(self,location_metrics):
        return [location_metrics[o,d] for o,d in self.from_to_tuples] 

    def volume_to_pickup_at(self,location):
        return sum(list(map(lambda cargo: cargo.volume, filter(lambda cargo: cargo.origin == location,self.iter_cargo))))
    
    def volume_to_delivery_at(self,location):
        return sum(list(map(lambda cargo: cargo.volume, filter(lambda cargo: cargo.destiny == location,self.iter_cargo))))

    def calculateMetrics(self,data):
        distance_matrix = data['distance_matrix']
        time_matrix = data['time_matrix']

        self.total_dist = 0
        self.total_time = 0
        self.times = []
        self.dists = []
        self.stops = len(self.from_to_tuples)
        current_volume = 0
        self.route_volumes = []
        for start,stop in self.from_to_tuples:
            self.times.append(time_matrix[start,stop])
            self.dists.append(distance_matrix[start,stop])
            volumeIn = self.volume_to_pickup_at(start)
            volumeOut = self.volume_to_delivery_at(stop)
            if (volumeIn-volumeOut+current_volume)>self.max_volume:
                self.listener(VehicleMaxVolumeExeeded(f"Max volume exedded in {self.name} on {self.code}({self.max_volume-volumeIn-volumeOut+current_volume})")) 
            if (volumeIn-volumeOut+current_volume)<0:
                self.listener(LogicError(f"Vehicle {self.code} negative volume.")) 

            if (volumeOut>current_volume):
                self.listener(LogicError(f"Vehicle {self.code} are delivering more cargo volume than it have:{current_volume} out:{volumeOut}")) 
            current_volume+=(volumeIn-volumeOut)
            self.route_volumes.append(current_volume)
        self.total_dist= np.array(self.dists).sum()
        self.total_time= np.array(self.times).sum()
        self.time_average = np.array(self.times).mean()
        self.dists_average = np.array(self.dists).mean()
        
        


        for cargo_code in self.__cargo: 
            cargo = self.__cargo[cargo_code]
            cargo.calculateMetrics(data)
        
    def printMetrics(self,distance_matrix,time_matrix):
        move=str(self.route[0])
        
        for i,(start,stop) in enumerate(self.from_to_tuples):
            move+="-({}m3|{}km|{}h)->{}".format(self.route_volumes[i],distance_matrix[start,stop],time_matrix[start,stop],stop)
        
        
        print("{}\t\t{}({}m3|{}km|{}h|{}|{}):\t{}".format(
            self.name,
            self.code, 
            self.max_volume,
            self.total_dist, 
            self.total_time, 
            self.stops,
            "S" if self.stopped else "R",
            move
            ))    

class Van(Vehicle):
    # 0 - Van:
    #       Volume:9 m3
    #       Capacidade de peso: 1.500 Quilos
    #       Comprimento: 2,95 m
    #       Largura: 1,70 m
    #       Altura: 1,80 m
    def __init__(self,vehicle_data):
        Vehicle.__init__(self,vehicle_data,vehicle_name="Van")
        self._Vehicle__volume = 9           # m3
        self._Vehicle__max_weight = 1500    # Kg
        self._Vehicle__lenght = 2.95        # m
        self._Vehicle__width = 1.7          # m
        self._Vehicle__height = 1.8         # m
        self._Vehicle__type_code = 0
    
class Medio(Vehicle):
    # 1 - Médio ou 3/4
    #       Volume 20 m3
    #       Capacidade de peso: 3.500 Quilos
    #       Comprimento: 4,45 m
    #       Largura: 2,10 m
    #       Altura: 2,14 m
    def __init__(self,vehicle_data):
        Vehicle.__init__(self,vehicle_data,vehicle_name="3/4")
        self._Vehicle__volume = 20          # m3
        self._Vehicle__max_weight = 3500    # Kg
        self._Vehicle__lenght = 4.5         # m
        self._Vehicle__width = 2,1          # m
        self._Vehicle__height = 2.14        # m
        self._Vehicle__type_code = 1

class VUC(Vehicle):
    # 2 - VUC: 
    #       Volume: 22 M3
    #       Capacidade de peso: 3.500 Quilos
    #       Comprimento: 4,30 m
    #       Largura: 2,30 m
    #       Altura: 2,20 m
 
    def __init__(self,vehicle_data):
        Vehicle.__init__(self,vehicle_data,vehicle_name="VUC")
        self._Vehicle__volume = 22 # m3
        self._Vehicle__max_weight = 3500 # Kg
        self._Vehicle__lenght = 4.3 #m
        self._Vehicle__width = 2.3
        self._Vehicle__height = 2.2 #m
        self._Vehicle__type_code = 2

class Toco(Vehicle):
    # 3 - Toco: 
    #       Volume: 46 M3
    #       Capacidade de peso: 7.000 Quilo
    #       Comprimento: 7,00 Metros
    #       Largura: 2,50 Metros
    #       Altura: 2,65 Metros
    def __init__(self,vehicle_data):
        Vehicle.__init__(self,vehicle_data,vehicle_name="Toco")
        self._Vehicle__volume = 46 # m3
        self._Vehicle__max_weight = 7000 # Kg
        self._Vehicle__lenght = 7.0 #m
        self._Vehicle__width = 2.5
        self._Vehicle__height = 2.65 #m
        self._Vehicle__type_code = 3

class Truck(Vehicle):
    # 4 - Truck: 
    #       Volume: 50 m3
    #       Capacidade de peso: 12.000 Quilos
    #       Comprimento: 7,50 Metros
    #       Largura: 2,50 Metros
    #       Altura: 2,70 Metros
    def __init__(self,vehicle_data):
        Vehicle.__init__(self,vehicle_data,vehicle_name="Truck")
        self._Vehicle__volume = 50 # m3
        self._Vehicle__max_weight = 12000 # Kg
        self._Vehicle__lenght = 7.5 #m
        self._Vehicle__width = 2.5
        self._Vehicle__height = 2.7 #m
        self._Vehicle__type_code = 4

class Carreta2(Vehicle):
    # 5 - Carreta 2 Eixos: 
    #       Volume: 95 M3
    #       Capacidade de peso: 18.000 Quilos
    #       Comprimento: 14,00 Metros
    #       Largura: 2,50 Metros
    #       Altura: 2,70 Metros
    def __init__(self,vehicle_data):
        Vehicle.__init__(self,vehicle_data,vehicle_name="Carreta")
        self._Vehicle__volume = 95 # m3
        self._Vehicle__max_weight = 18000 # Kg
        self._Vehicle__lenght = 14.0 #m
        self._Vehicle__width = 2.5
        self._Vehicle__height = 2.7 #m
        self._Vehicle__type_code = 5

class Carreta3(Vehicle):
    # 6 - Carreta 3 Eixos: 
    #       Volume: 95 M3
    #       Capacidade de peso:25.000 Quilos
    #       Comprimento: 14,00 Metros
    #       Largura: 2,50 Metros
    #       Altura: 2,50 metros
    def __init__(self,vehicle_data):
        Vehicle.__init__(self,vehicle_data,vehicle_name="Carreta 3 eixos")
        self._Vehicle__volume = 95 # m3
        self._Vehicle__max_weight = 25000 # Kg
        self._Vehicle__lenght = 14.0 #m
        self._Vehicle__width = 2.5
        self._Vehicle__height = 2.5 #m
        self._Vehicle__type_code = 6

class CargoManager(object):
    """
    Usage:
        manager = CargoManager(data)
    
    Data must have: 
        - distance_matrix
        - vehicle_data 
        - cargo

    See: VRP.create_data_model_small()
    """
    
    def __init__(self,data):
        self.__cargo_routes = None
        self.__movements = None
        self.__optimization_exceptions = []
        self.__data = data
        self.__vehicles = {}
        self.__cargos = {}
        self.cargo = list(map(lambda cargo_data: Cargo(cargo_data), data['cargo']))
        self.max_locations  = data['distance_matrix'].shape[0]
        self.vehicles_count   = len(data['vehicle_data'])
        self.vehicle_locations = data['vehicle_data']
        self.vehicle_type_map = {
            0:Van,
            1:Medio,
            2:VUC,
            3:Toco,
            4:Truck,
            5:Carreta2,
            6:Carreta3
        }
        
        self.total_only_on_low_resources_penalty=0
        self.total_prohibited_exception_penalty=0

        self.total_not_recomended_exception_penalty=0
        self.count_of_logistics_error=0
        self.count_of_logic_error=0
        self.count_of_origin_NonExistent=0
        self.count_of_destiny_NonExistent=0
        self.same_location_count=0
        self.time_average = 0
        self.dist_average = 0
        self.moves_average = 0

    @property
    def movements(self):
        return self.__movements
    
    def __create_vehicle(self,vehicle_data,route):
        vehicle = self.vehicle_type_map[vehicle_data[0]](vehicle_data)
        vehicle.route = route
        return vehicle
    @movements.setter
    def movements(self,movements):
        self.__movements = movements
        self.vehicles = []
        # self.vehicles = list(map(lambda vehicle_data: ,))
        for route,vehicle_data in zip(self.__movements,self.__data['vehicle_data']):
            self.vehicles.append(self.__create_vehicle(vehicle_data,route))
            
    
    @property
    def cargo_routes(self):
        return self.__cargo_routes
    
    @cargo_routes.setter
    def cargo_routes(self,cargo_routes):
        self.__cargo_routes = cargo_routes


    @property
    def stopped_vehicles(self): 
        return list(filter(lambda vehicle: vehicle.stopped, self.vehicles))
    
    @property
    def running_vehicles(self): 
        return list(filter(lambda vehicle: not vehicle.stopped, self.vehicles))
    
    @property
    def exceptions(self):
        return ",\n".join(["{}: {}".format(type(ex).__name__,str(ex)) for ex in self.iter_optimization_exceptions])
    
    @property
    def optimization_penalty(self):
        return sum([ex.penalty for ex in self.iter_optimization_exceptions])
    
    @property
    def total_optimization_penalty(self):
        return sum([ex.penalty for ex in self.iter_optimization_exceptions])

    @property
    def total_logistic_penalty(self):
        return sum([ex.penalty for ex in filter(lambda ex: isinstance(ex,LogisticException),self.iter_optimization_exceptions)])

    @property
    def total_logic_penalty(self):
        return sum([ex.penalty for ex in filter(lambda ex: isinstance(ex,LogicError),self.iter_optimization_exceptions)])

    @property
    def optimization_exception_count(self):
        return len(list(self.iter_optimization_exceptions))
    
    @property
    def iter_optimization_exceptions(self):
        for ex in self._CargoManager__optimization_exceptions:
            yield ex

    @property
    def total_space_left(self):
        return sum([ vehicle.max_volume-vehicle.current_volume for vehicle in self.iter_vehicle])
    
    @property
    def total_volume_unalocated(self):
        total_space_left = self.total_space_left
        return sum([cargo.volume for cargo in filter(lambda cargo: (not cargo.pickuped) & (cargo.volume<=total_space_left), self.iter_cargo)])

    @property
    def space_not_alocated_penalty(self):
        space_not_alocated = self.total_space_left
        return  space_not_alocated * self.total_volume_unalocated

    def registerException(self,ex):
        self._CargoManager__optimization_exceptions.append(ex)

    def listener(self,ex):
        try:
            raise ex
        except OptimizationException as ex:
            self.registerException(ex)

    def registerCargo(self,cargo,vehicle):
        vehicle.listener = self.listener
        cargo.listener = self.listener
        if vehicle.code in self.__vehicles:
            vehicle = self.__vehicles[vehicle.code]
        else: 
            self.__vehicles[vehicle.code] = vehicle
        vehicle.addCargo(cargo)
    
    def evaluate(self,debug=None):
        for cargo,vehicle in enumerate(self.cargo_routes):
            
                if vehicle >= len(self.vehicles):
                    self.listener(LogicError("There is no car for route {} ".format(vehicle)))
                    continue
                self.registerCargo(self.cargo[cargo],self.vehicles[vehicle])
            
            
        self.calculateMetrics()
    
    @property
    def iter_cargo(self):
        for cargo in self.cargo:
            yield cargo
    
    @property
    def iter_vehicle(self):
        for vehicle in self.__vehicles:
            yield self.__vehicles[vehicle]

    def penaltyByExceptionType(self,optimization_exception_type):
        return sum([ex.penalty for ex in filter(lambda ex: isinstance(ex,optimization_exception_type),self.iter_optimization_exceptions)])

    def countByExceptionType(self,optimization_exception_type):
        return len(list(filter(lambda ex: isinstance(ex,optimization_exception_type),self.iter_optimization_exceptions)))

    def calculateMetrics(self):
        self.total_dist=0
        self.total_time=0
        self.total_moves=0
        self.routes_dist = []
        self.routes_time = []
        self.routes_moves = []
        routes_distances_times = []

        for vehicle in self.vehicles:
            vehicle.calculateMetrics(self.__data)
            
            self.routes_dist.append(vehicle.total_dist)
            self.routes_time.append(vehicle.total_time)
            self.routes_moves.append(vehicle.stops)
        self.total_dist=np.array(self.routes_dist).sum()
        self.total_time=np.array(self.routes_time).sum()
        self.total_moves=np.array(self.routes_moves).sum()
        
        self.time_average=np.array(self.routes_time).mean()
        self.dist_average=np.array(self.routes_dist).mean()
        self.moves_average=np.array(self.routes_moves).mean()
            

def genomeToValues(genome,data):
    movements = genomeToMovements(
        genome[:data['genome_bit_count_routes']],data)
    cargo_routes = genomeToCargoRoutes(
        genome[data['genome_bit_count_routes']:data['genome_bit_count_routes']+data['genome_bit_count_cargo']],
        data['cargo_count'])
    return movements,cargo_routes

def initpop(genomesize,population_size,dtype):
    pop = np.load('vrp_pop.npy')
    print('Loaded from file')
    return pop 

def car_stopped_count(movements):
    return sum(sum(np.diff(movements,axis=1)==0)[:0])

def ciclical_route_policy(movements,vehicle_locations):
    return sum(np.array([i for i in ciclical_route(np.concatenate((vehicle_locations,movements),axis=1))]))
        
def ciclical_route(movements):
    for i in range(movements.shape[0]):
        unique, counts = np.unique(movements[i], return_counts=True)
        yield sum(counts[(counts>1) & (unique>=0)])

def iter_routes(movements,vehicle_locations=None):
    '''
        Used to yield an tuple with movements like (from,to)
    '''
    for i in range(1,movements.shape[0]):
        for origin,destiny in moviment_tuple(np.concatenate(([vehicle_locations[i]],movements[i]))):
            yield origin,destiny

def moviment_tuple(movements):
    for j in range(movements.shape[0]-1):
        if(movements[j]>=0) & (movements[j+1]>=0):
            if(movements[j+1]==movements[j]):
                continue
            yield (movements[j],movements[j+1])
        else: 
            break

def genomeToMovements(genome,data):
    locations  = data['distance_matrix'].shape[0]
    vehicles   = len(data['vehicle_data'])
    steps = data['max_steps']
    
    bits = genome.reshape((steps*vehicles,-1)) 
    movement = binary_ops.bitsToBytes(bits)
    movement = np.ndarray.astype(movement.reshape((vehicles,-1)),copy=True,dtype=int)
    movement[movement >= locations] = locations
    return movement-1

def genomeToCargoRoutes(genome,routes):
    bits = genome.reshape((routes,-1)) 
    cargo_routes = binary_ops.bitsToBytes(bits)
    return cargo_routes

def gaConfig(genomesize,data):
    
    
    # population_size = 100
    # population = initpop() * np.ones((population_size,1))

    ga = GeneticAlgorithm.GA(genomesize,
        population_size=200,
        # population=initpop,
        epochs=1000,
        ephoc_generations=10,
        selection_count=50,
        maximization=False,
        on_ephoc_ends=lambda _,genome: evaluate(genome,data)
    )
    
    ga.debug = False
    ga.verbose = True
    return ga

def create_data_model_small():
    """
        Gera dados de 17 localizações e cargas de forma prefixada.

    
        data['distance_matrix'] = Distancia entre as localizações (Km).
            Definido por uma relação de equivalência entre o tempo e uma 
            velocidade fixa de 90 km/h

        data['time_matrix'] = Definição do tempo em horas (h) entre duas
            localizações

        data['time_windows'] = Definição da janela de entrega 
            para cada encomenda.

    """
    data = {}

   
    data['time_matrix'] = np.array([
        [ 0, 6,  9,  8 ],
        [ 6, 0,  8,  3 ],
        [ 9, 8,  0,  11],
        [ 8, 3,  11, 0 ],
      ])

    
    data['distance_matrix'] = data['time_matrix']*90 

    # tipo
    # 0 - Van:
    #       Volume:9 m3
    #       Capacidade de peso: 1.500 kg
    #       Comprimento: 1,70 m
    #       Altura: 1,80 m
    # 1 - Médio ou 3/4
    #       Volume 20 m3
    #       Capacidade de peso: 3.500 Quilos
    #       Comprimento: 4,45 m
    #       Largura: 2,10 m
    #       Altura: 2,14 m
    # 2 - VUC: 
    #       Volume: 22 M3
    #       Capacidade de peso: 3.500 Quilos
    #       Comprimento: 4,30 m
    #       Largura: 2,30 m
    #       Altura: 2,20 m
    # 3 - Toco: 
    #       Volume: 46 M3
    #       Capacidade de peso: 7.000 Quilo
    #       Comprimento: 7,00 Metros
    #       Largura: 2,50 Metros
    #       Altura: 2,65 Metros
    # 4 - Truck: 
    #       Volume: 50 m3
    #       Capacidade de peso: 12.000 Quilos
    #       Comprimento: 7,50 Metros
    #       Largura: 2,50 Metros
    #       Altura: 2,70 Metros
    #     
    # 5 - Carreta 2 Eixos: 
    #       Volume: 95 M3
    #       Capacidade de peso: 18.000 Quilos
    #       Comprimento: 14,00 Metros
    #       Largura: 2,50 Metros
    #       Altura: 2,70 Metros
    # 6 - Carreta 3 Eixos: 
    #       Volume: 95 M3
    #       Capacidade de peso:25.000 Quilos
    #       Comprimento: 14,00 Metros
    #       Largura: 2,50 Metros
    #       Altura: 2,50 metros
    #
    
    #type,id,location_id
    data['vehicle_data'] = np.array([[1,223,0],
                                     [2,472,2],
                                     [3,897,3],
                                     [4,991,1],
                                     [5,890,1]])

    # - Bitrem 
    # - Carreta (Peso maximo 30-45 Ton)
    # -     Tipos
    #           Baú ( Consegue carregar caixa solta - "carga batida" - n eh o caso da CL)
    #           Sider <<- maior parte ( Carga pela lateral é mais rápido - precisa ser pallet)
    #           Lonadas ->> eqq Graneleiros
    # - Truck
    # - Toco
    


    # Carga
    #    Tipos: 
    #       - Pallet (madeira)
    #       - Rach (Pallet de Metal - Vira tipo uma gaiola)
    
    # origin_id,dest_id,cargo_id,volume(m3),max_stack,cubagem(kg/m3 - Densidade),peso(kg),tempo_min,tempo_max
    data['cargo'] = np.array(
      [[1, 2, 443, 3, 3],
       [2, 3, 229, 3, 1],
       [0, 3, 878, 1, 1],
       [2, 1, 332, 1, 2],
       [2, 1, 576, 1, 2],
       [2, 1, 942, 1, 2],
       [3, 0, 417, 1, 2]]
    )
    data['max_steps']=5
    return data

def create_data_model_cargo_stress():
    """
        Gera dados de 17 localizações e cargas de forma prefixada.

    
        data['distance_matrix'] = Distancia entre as localizações (Km).
            Definido por uma relação de equivalência entre o tempo e uma 
            velocidade fixa de 90 km/h

        data['time_matrix'] = Definição do tempo em horas (h) entre duas
            localizações

        data['time_windows'] = Definição da janela de entrega 
            para cada encomenda.

    """
    data = {}

   
    data['time_matrix'] = np.array([
        [ 0, 6,  9,  8 ],
        [ 6, 0,  8,  3 ],
        [ 9, 8,  0,  11],
        [ 8, 3,  11, 0 ],
      ])

    
    data['distance_matrix'] = data['time_matrix']*90 

    # tipo
    # 0 - Van:
    #       Volume:9 m3
    #       Capacidade de peso: 1.500 kg
    #       Comprimento: 1,70 m
    #       Altura: 1,80 m
    # 1 - Médio ou 3/4
    #       Volume 20 m3
    #       Capacidade de peso: 3.500 Quilos
    #       Comprimento: 4,45 m
    #       Largura: 2,10 m
    #       Altura: 2,14 m
    # 2 - VUC: 
    #       Volume: 22 M3
    #       Capacidade de peso: 3.500 Quilos
    #       Comprimento: 4,30 m
    #       Largura: 2,30 m
    #       Altura: 2,20 m
    # 3 - Toco: 
    #       Volume: 46 M3
    #       Capacidade de peso: 7.000 Quilo
    #       Comprimento: 7,00 Metros
    #       Largura: 2,50 Metros
    #       Altura: 2,65 Metros
    # 4 - Truck: 
    #       Volume: 50 m3
    #       Capacidade de peso: 12.000 Quilos
    #       Comprimento: 7,50 Metros
    #       Largura: 2,50 Metros
    #       Altura: 2,70 Metros
    #     
    # 5 - Carreta 2 Eixos: 
    #       Volume: 95 M3
    #       Capacidade de peso: 18.000 Quilos
    #       Comprimento: 14,00 Metros
    #       Largura: 2,50 Metros
    #       Altura: 2,70 Metros
    # 6 - Carreta 3 Eixos: 
    #       Volume: 95 M3
    #       Capacidade de peso:25.000 Quilos
    #       Comprimento: 14,00 Metros
    #       Largura: 2,50 Metros
    #       Altura: 2,50 metros
    #
    
    #type,id,location_id
    data['vehicle_data'] = np.array([[1,223,0],
                                     [2,472,2],
                                     [3,897,3],
                                     [4,991,1],
                                     [5,890,1]])

    # - Bitrem 
    # - Carreta (Peso maximo 30-45 Ton)
    # -     Tipos
    #           Baú ( Consegue carregar caixa solta - "carga batida" - n eh o caso da CL)
    #           Sider <<- maior parte ( Carga pela lateral é mais rápido - precisa ser pallet)
    #           Lonadas ->> eqq Graneleiros
    # - Truck
    # - Toco
    


    # Carga
    #    Tipos: 
    #       - Pallet (madeira)
    #       - Rach (Pallet de Metal - Vira tipo uma gaiola)
    
    # origin_id,dest_id,cargo_id,volume(m3),max_stack,cubagem(kg/m3 - Densidade),peso(kg),tempo_min,tempo_max
    cargo_count = 100
    data['cargo'] = np.concatenate(
        (np.random.randint(0,4,(cargo_count,1)),np.random.randint(0,4,(cargo_count,1)), #origin-destiny
        np.random.choice(range(100,1000),(cargo_count,1),replace=False), #cargo-id
        np.random.randint(1,4,(cargo_count,1))),axis=1)
    data['max_steps'] = data['distance_matrix'].shape[0]
    return data

def fitness(genome,data,cargo_manager=None,verbose=False):
    c = 0.001
    
    if cargo_manager is None:
        manager = CargoManager(data)
    else: 
        manager = cargo_manager
    
    
    manager.movements,manager.cargo_routes = genomeToValues(genome,data)
    manager.evaluate()
    score = 0
    if len(genome) != data['genome_bit_count_routes']+data['genome_bit_count_cargo']:
        # self.listener(LogicError('Something is wrong genome_bit_count_routes+genome_bit_count_cargo must be equals genome size'))
        score+=10**10




    #Low priority Optimization  0.001
    time_penalty = 0.001 * (manager.total_time/max(c,manager.total_moves))
    distance_penalty = 0.001 * (manager.total_dist/max(c,manager.total_moves))
    

    # Average optimisation scale 10
    scale = 0.0000001
    average_time_penalty = scale * manager.time_average
    average_dist_penalty = scale * manager.dist_average
    average_moves_penalty = scale * manager.moves_average

    # Somente em caso de recurso escasso  scale 1k
    # scale = 10000
    origin_NonExistent_penalty =  scale * manager.count_of_origin_NonExistent
    destiny_NonExistent_penalty =  scale * manager.count_of_destiny_NonExistent
    same_location_penalty =   scale * manager.same_location_count
    
    car_stopped_count = len(manager.stopped_vehicles)
    vehicle_stopped_penalty = \
        ((c+car_stopped_count) * destiny_NonExistent_penalty) + \
        ((c+car_stopped_count) * origin_NonExistent_penalty)
    
    
    

    
    if verbose: 
        print("Total Distance: {}Km".format(manager.total_dist))
        print("Total Time: {}h".format(manager.total_time))
        print("Total Stops: {}".format(manager.total_moves))
        
        print("Distance Penalty:",distance_penalty)
        print("Time Penalty:",time_penalty)
        
        print("Same Location Penalty:".format(
            manager.countByExceptionType(SameLocationException),
            manager.penaltyByExceptionType(SameLocationException)))
        
        print("Logistic Penalty ({}): {}".format(
            manager.countByExceptionType(LogisticException),
            manager.penaltyByExceptionType(LogisticException)))
        
        print("Origin Penalty ({}): {}".format(
            manager.countByExceptionType(OriginNonExistentException),
            manager.penaltyByExceptionType(OriginNonExistentException)))
        
        print("Destiny NonExistent({}): {}".format(
            manager.countByExceptionType(DestinyNonExistentException),
            manager.penaltyByExceptionType(DestinyNonExistentException)))
        
        print("Logic Penalty ({}): {}".format(
            manager.countByExceptionType(LogicError),
            manager.penaltyByExceptionType(LogicError)))
        
        print("Stopped Penalty ({}): {}".format(
            manager.countByExceptionType(VehicleStoppedException),
            manager.penaltyByExceptionType(VehicleStoppedException)))
        
        print("Vehicle Volume Exeeded ({}): {}".format(
            manager.countByExceptionType(VehicleMaxVolumeExeeded),
            manager.penaltyByExceptionType(VehicleMaxVolumeExeeded)))
        
        print("Space Left({}): {}".format(manager.total_space_left,manager.space_not_alocated_penalty))
        
        print("Time average:",manager.time_average) 
        print("Dist average:",manager.dist_average) 
        print("Moves average:",manager.moves_average) 


    score = score \
        + distance_penalty \
        + time_penalty \
        + average_time_penalty \
        + average_dist_penalty \
        + average_moves_penalty \
        + manager.space_not_alocated_penalty \
        + manager.total_optimization_penalty \
   

    return score

def main():
    data = create_data_model_small()
    # data = create_data_model_cargo_stress()
    locations  = data['distance_matrix'].shape[0]
    vehicles   = len(data['vehicle_data'])
    max_steps = data['max_steps']
    cargos     = data['cargo'].shape[0]
    
    data['vehicle_count'] = vehicles
    data['cargo_count'] = cargos

    
    locations_byte_size  = binary_ops.bitsNeededToNumber(locations)
    cargo_byte_count  = binary_ops.bitsNeededToNumber(vehicles)
    
    genome_bit_count_routes = locations_byte_size * vehicles * max_steps
    data['genome_bit_count_routes'] = genome_bit_count_routes
    
    genome_bit_count_cargo = cargo_byte_count*cargos
    data['genome_bit_count_cargo'] = genome_bit_count_cargo
    

    genome_size = genome_bit_count_routes+genome_bit_count_cargo
    
    print("==========Initialization==========")
    print("- Using genome Size: ",genome_size)
    print("- genome Byte Size: ", locations_byte_size )
    print("- Locations: ",locations)
    print("- Vehicles: ",vehicles)
    print("- Max Steps: ",vehicles)
    
    ga = gaConfig(genome_size,data)
    best, pop, scores = \
        ga.run(lambda genome: fitness(genome,data))

    
    # evaluate(best,data)

    np.save('vrp_pop', pop)
    

def evaluate(genome,data):
    manager = CargoManager(data)
    # movements = manager.movements
    # cargo_routes = manager.cargo_routes
    manager.movements,manager.cargo_routes = genomeToValues(genome,data)
    manager.evaluate()
    manager.calculateMetrics()
    

    print('============Routes==============')
    for cargo in manager.iter_cargo:
        cargo.printMetrics()
    
    print('=============DATA================')
    print("Cargos: ")
    print(data['cargo'])
    print("Vehicles Location:")
    print(data['vehicle_data'][:,1])
    print("Vehicles Code:")
    print(data['vehicle_data'][:,0])
    print("Distances: ")
    print(data['distance_matrix'])
    print('============Manager=============')
    print(manager.exceptions)
    print('==========Optimization==========')
    print("Cargo Routes:")
    print(manager.cargo_routes)
    print("Movements:")
    print(manager.movements)
    
    print('============Metrics=============')
    print("Total Penalty:",fitness(genome,data,cargo_manager=manager,verbose=True))
    print('============Vehicles============')
    for vehicle in manager.vehicles:
        vehicle.printMetrics(data['distance_matrix'],data['time_matrix'])
    
        
    print('================================')


if __name__ == '__main__':
    main()




    