"""
A simple test script to verify the SchedulingAlgorithm class works.
"""

from scheduling_algorithm import SchedulingAlgorithm

def main():
    print("Creating a scheduler with default values (3 machines, 16 pockets):")
    scheduler1 = SchedulingAlgorithm()
    print(f"  machines_count: {scheduler1.machines_count}")
    print(f"  pockets_per_machine: {scheduler1.pockets_per_machine}")
    print(f"  machine_names: {scheduler1.machine_names}")
    
    print("\nCreating a scheduler with custom values (5 machines, 20 pockets):")
    scheduler2 = SchedulingAlgorithm(machines_count=5, pockets_per_machine=20)
    print(f"  machines_count: {scheduler2.machines_count}")
    print(f"  pockets_per_machine: {scheduler2.pockets_per_machine}")
    print(f"  machine_names: {scheduler2.machine_names}")
    
    print("\nVerification complete!")

if __name__ == "__main__":
    main() 