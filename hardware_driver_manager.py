"""
Hardware driver manager - loads and instantiates drivers.

This module provides the HardwareDriverManager class which handles:
- Loading hardware profiles from JSON files
- Dynamically importing and instantiating driver classes
- Managing the currently active driver
"""

import json
import importlib
from pathlib import Path
from typing import Dict, List, Optional
from hardware_driver import HardwareDriver, HardwareProfile


class HardwareDriverManager:
    """
    Manages hardware driver profiles and instantiation.
    
    This class scans a directory for JSON profile files, loads them,
    and provides factory methods to create driver instances.
    """
    
    def __init__(self, profiles_dir: str = "hardware_profiles"):
        """
        Initialize driver manager.
        
        Args:
            profiles_dir: Directory containing profile JSON files
        """
        self.profiles_dir = Path(profiles_dir)
        self.profiles: Dict[str, HardwareProfile] = {}
        self.current_driver: Optional[HardwareDriver] = None
        
        self.load_profiles()
    
    def load_profiles(self) -> None:
        """Load all hardware profiles from JSON files."""
        if not self.profiles_dir.exists():
            self.profiles_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created profiles directory: {self.profiles_dir}")
            return
        
        for profile_file in self.profiles_dir.glob("*.json"):
            try:
                profile = self.load_profile(profile_file)
                self.profiles[profile.name] = profile
                print(f"Loaded profile: {profile.name}")
            except Exception as e:
                print(f"Error loading {profile_file.name}: {e}")
    
    def load_profile(self, filepath: Path) -> HardwareProfile:
        """
        Load a single hardware profile from JSON.
        
        Args:
            filepath: Path to profile JSON file
            
        Returns:
            HardwareProfile object
            
        Raises:
            ValueError: If required fields are missing
            JSONDecodeError: If file is not valid JSON
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Validate required fields
        if 'name' not in data:
            raise ValueError(f"Profile missing 'name' field: {filepath.name}")
        if 'driver' not in data:
            raise ValueError(f"Profile missing 'driver' field: {filepath.name}")
        
        return HardwareProfile(
            name=data['name'],
            driver_class=data['driver'],
            baudrate=data.get('baudrate', 115200),
            timeout=data.get('timeout', 1.0),
            terminator=data.get('terminator', '\n'),
            commands=data.get('commands', {}),
            data_format=data.get('data_format', {})
        )
    
    def get_profile_names(self) -> List[str]:
        """
        Get list of available profile names.
        
        Returns:
            Sorted list of profile names
        """
        return sorted(self.profiles.keys())
    
    def get_profile(self, profile_name: str) -> Optional[HardwareProfile]:
        """
        Get a specific profile by name.
        
        Args:
            profile_name: Name of the profile
            
        Returns:
            HardwareProfile object, or None if not found
        """
        return self.profiles.get(profile_name)
    
    def create_driver(self, profile_name: str) -> Optional[HardwareDriver]:
        """
        Instantiate a driver from a profile.
        
        This method dynamically imports the driver class specified in the profile
        and creates an instance with the profile configuration.
        
        Args:
            profile_name: Name of the hardware profile
            
        Returns:
            HardwareDriver instance, or None if creation failed
        """
        if profile_name not in self.profiles:
            print(f"Profile '{profile_name}' not found")
            return None
        
        profile = self.profiles[profile_name]
        
        try:
            # Convert class name to module name
            # Example: GenericSerialDriver -> generic_serial_driver
            module_name = f"drivers.{self._class_to_module(profile.driver_class)}"
            
            # Dynamically import the module
            module = importlib.import_module(module_name)
            
            # Get the driver class from the module
            driver_class = getattr(module, profile.driver_class)
            
            # Instantiate and return
            driver = driver_class(profile)
            print(f"Created driver: {profile.driver_class} for {profile.name}")
            return driver
            
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(f"Driver module not found {module_name}")
        except AttributeError as e:
            raise AttributeError(f"Driver class '{profile.driver_class}' not found in module")
        except Exception as e:
            raise Exception(f"Error creating driver: {e}")
    
    @staticmethod
    def _class_to_module(class_name: str) -> str:
        """
        Convert class name to module filename.
        
        Uses snake_case convention for module names.
        
        Args:
            class_name: CamelCase class name
            
        Returns:
            snake_case module name
            
        Example:
            CustomADCDriver -> custom_adc_driver
            GenericSerialDriver -> generic_serial_driver
        """
        import re
        # Insert underscore before capitals (except at start), then lowercase
        module = re.sub('(?<!^)(?=[A-Z])', '_', class_name).lower()
        return module
    
    def set_current_driver(self, driver: HardwareDriver) -> None:
        """
        Set the currently active driver.
        
        Args:
            driver: HardwareDriver instance to set as current
        """
        self.current_driver = driver
    
    def get_current_driver(self) -> Optional[HardwareDriver]:
        """
        Get the currently active driver.
        
        Returns:
            Current HardwareDriver instance, or None if no driver is active
        """
        return self.current_driver
    
    def reload_profiles(self) -> None:
        """
        Reload all profiles from the profiles directory.
        
        Useful for picking up changes to profile files without restarting.
        """
        self.profiles.clear()
        self.load_profiles()
    
    def save_profile(self, profile: HardwareProfile, filename: Optional[str] = None) -> bool:
        """
        Save a hardware profile to JSON file.
        
        Args:
            profile: HardwareProfile to save
            filename: Optional filename (defaults to profile name)
            
        Returns:
            True if save successful, False otherwise
        """
        if filename is None:
            # Convert profile name to valid filename
            filename = profile.name.lower().replace(' ', '_') + '.json'
        
        filepath = self.profiles_dir / filename
        
        try:
            data = {
                'name': profile.name,
                'driver': profile.driver_class,
                'baudrate': profile.baudrate,
                'timeout': profile.timeout,
                'terminator': profile.terminator,
                'commands': profile.commands,
                'data_format': profile.data_format
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
            
            print(f"Saved profile to: {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving profile: {e}")
            return False
