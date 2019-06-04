import unittest
import numpy as np
from ..VRP import Van,Medio,VUC,Toco,Truck,Carreta2,Carreta3
from ..VRP import Cargo,Vehicle,CargoManager
from ..VRP import VehicleMaxVolumeExeeded,OriginNonExistentException,DestinyNonExistentException

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


class VehicleLogistics(unittest.TestCase):
    def setUp(self):   

        self.manager = CargoManager(create_data_model_small())

    def test_SameLocationRoute(self):
        vehicles = [
            Van([100,1]) 
            ,Medio([100,1]) 
            ,VUC([100,1]) 
            ,Toco([100,1]) 
            ,Truck([100,1]) 
            ,Carreta2([100,1]) 
            ,Carreta3([100,1]) 
        ]
        exceptions = []
        def listener(ex):
            exceptions.append(ex)
            
        for vehicle in vehicles:
            vehicle.listener = listener
            vehicle.route = [2,2,2]
        
        expected = (1,2)
        received = vehicle.from_to_tuples[0]
        assert len(vehicle.from_to_tuples) == 0
        assert expected[0] == received[0]
        assert expected[1] == received[1]

    
        
    def test_VehicleMaxVolumeExeeded(self): 
        #         Van([identifier,location])
        vehicles = [
            Van([1,100,1]) # 9 max volume
            ,Medio([1,100,1]) # 20 max volume
            ,VUC([1,100,1]) # 20 max volume
            ,Toco([1,100,1]) # 46 max volume
            ,Truck([1,100,1]) # 50 max volume
            ,Carreta2([1,100,1]) # 90 max volume
            ,Carreta3([1,100,1]) # 95 max volume
        ]
        
        
        exceptions = []
        def listener(ex):
            exceptions.append(ex)
        for vehicle in vehicles:
            cargo = Cargo([1, 2, 229, 100, 1]) # 100 should not fit in any vehicle
            
            cargo.listener = listener
            vehicle.listener = listener

            vehicle.route = [1,2]
            vehicle.addCargo(cargo)
            # vehicle.calculateMetrics()
        
        self.assertEqual(len(exceptions),7) 
        for ex in exceptions:
            assert(isinstance(ex,VehicleMaxVolumeExeeded))

    # def test_DestinyNonExistentException(self):
    #     raise Exception("not implemented")

    # def test_OriginNonExistentException(self):
        
    #     van = Van([100,1])
    #     van.route = [1,2,3,4]
    #     cargo = Cargo([6, 1, 229, 100, 1]) # 100 should not fit in any vehicle

    #     exceptions = []
    #     def listener(ex):
    #         exceptions.append(ex)
    #     van.addCargo(cargo)
    # def test_VehicleStoppedException(self):
    #     raise Exception("not implemented")
    
    # def test_cargo_manager_listener(self):
    #     raise Exception("not implemented")
    
    # def test_vehicle_arrived_in_time(self):
    #     raise Exception("not implemented")


class CargoTests(unittest.TestCase):

    def setUp(self):
        class MockVehicle(Vehicle):
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

        self.vehicle = MockVehicle([1,1,1,1,1,1,1])
        self.vehicle.route = [0,3,-1,0] # route: 0->3

        self.origin = 1
        self.dest = 2
        self.code = 229
        self.volume = 100
        self.maxstack = 1
        self.exceptions = []
        self.cargo1 = Cargo([self.origin, 3, self.code, self.volume, self.maxstack]) # 100 should not fit in any vehicle
        self.cargo2 = Cargo([0, self.dest, self.code, self.volume, self.maxstack]) # 100 should not fit in any vehicle
        
        self.listener = lambda ex: self.exceptions.append(ex)
        
        self.cargo1.listener = self.listener
        self.cargo2.listener = self.listener
        self.vehicle.listener = self.listener
        self.vehicle.addCargo(self.cargo)
        
    def test_property(self):
        self.assertEqual(self.cargo.volume      , self.volume)
        self.assertEqual(self.cargo.origin      , self.origin)
        self.assertEqual(self.cargo.destiny     , self.dest)
        self.assertEqual(self.cargo.code        , self.code)
        self.assertEqual(self.cargo.max_stack   , self.maxstack)

    def test_set_vehicle(self):
        self.assertEqual(len(self.exceptions), 2)
    def test_set_vehicle_exception_1(self):
        self.assertIsInstance(self.exceptions[0],DestinyNonExistentException)
    def test_set_vehicle_exception_2(self):
        self.assertIsInstance(self.exceptions[1],OriginNonExistentException)

        
        
        
                        


