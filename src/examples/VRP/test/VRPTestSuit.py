import unittest
from ..VRP import Van,Medio,VUC,Toco,Truck,Carreta2,Carreta3
from ..VRP import Cargo,Vehicle
from ..VRP import VehicleMaxVolumeExeeded,OriginNonExistentException,DestinyNonExistentException

class VehicleLogistics(unittest.TestCase):

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
            Van([100,1]) # 9 max volume
            ,Medio([100,1]) # 20 max volume
            ,VUC([100,1]) # 20 max volume
            ,Toco([100,1]) # 46 max volume
            ,Truck([100,1]) # 50 max volume
            ,Carreta2([100,1]) # 90 max volume
            ,Carreta3([100,1]) # 95 max volume
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
            vehicle.calculateMetrics()
        
        assert len(exceptions)>0 
        for ex in exceptions:
            assert(isinstance(ex,VehicleMaxVolumeExeeded))

    def test_DestinyNonExistentException(self):
        raise Exception("not implemented")

    def test_OriginNonExistentException(self):
        
        van = Van([100,1])
        van.route = [1,2,3,4]

        exceptions = []
        def listener(ex):
            exceptions.append(ex)
        van.listener 
  
    def test_VehicleStoppedException(self):
        raise Exception("not implemented")
    
    def test_cargo_manager_listener(self):
        raise Exception("not implemented")
    
    def test_vehicle_arrived_in_time(self):
        raise Exception("not implemented")


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
        self.cargo = Cargo([self.origin, self.dest, self.code, self.volume, self.maxstack]) # 100 should not fit in any vehicle
        self.listener = lambda ex: self.exceptions.append(ex)
        self.cargo.listener = self.listener
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
        self.assertIsInstance(self.exceptions[0],OriginNonExistentException)
    def test_set_vehicle_exception_2(self):
        self.assertIsInstance(self.exceptions[1],DestinyNonExistentException)

        
        
        
                        


