"""
Hardware driver manager: loads profiles from JSON and instantiates drivers.
"""

import json
import importlib
import re
from pathlib import Path
from typing import Dict, List, Optional

from hardware.base import HardwareDriver, HardwareProfile


class HardwareDriverManager:
    """
    Manages hardware driver profiles and instantiation.

    Scans a directory for JSON profile files, loads them, and provides
    factory methods to create driver instances dynamically.
    """

    def __init__(self, profiles_dir: str = "hardware_profiles") -> None:
        self.profiles_dir = Path(profiles_dir)
        self.profiles: Dict[str, HardwareProfile] = {}
        self.current_driver: Optional[HardwareDriver] = None
        self.load_profiles()

    # ─── Profile loading ──────────────────────────────────────────────────

    def load_profiles(self) -> None:
        """Load all hardware profiles from the profiles directory."""
        if not self.profiles_dir.exists():
            self.profiles_dir.mkdir(parents=True, exist_ok=True)
            return

        for profile_file in self.profiles_dir.glob("*.json"):
            try:
                profile = self.load_profile(profile_file)
                self.profiles[profile.name] = profile
                print(f"Loaded profile: {profile.name}")
            except Exception as e:
                print(f"Error loading {profile_file.name}: {e}")

    def load_profile(self, filepath: Path) -> HardwareProfile:
        """Load a single hardware profile from a JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        if "name" not in data:
            raise ValueError(f"Profile missing 'name' field: {filepath.name}")
        if "driver" not in data:
            raise ValueError(f"Profile missing 'driver' field: {filepath.name}")

        return HardwareProfile(
            name=data["name"],
            driver_class=data["driver"],
            baudrate=data.get("baudrate", 115200),
            timeout=data.get("timeout", 1.0),
            terminator=data.get("terminator", "\n"),
            return_on_init=data.get("return_on_init"),
            commands=data.get("commands", {}),
            data_format=data.get("data_format", {}),
        )

    def reload_profiles(self) -> None:
        """Reload all profiles from disk."""
        self.profiles.clear()
        self.load_profiles()

    # ─── Driver factory ───────────────────────────────────────────────────

    def create_driver(self, profile_name: str) -> Optional[HardwareDriver]:
        """
        Instantiate a driver from a profile.

        Dynamically imports the driver module under ``hardware.drivers``
        using the class name from the profile.
        """
        if profile_name not in self.profiles:
            print(f"Profile '{profile_name}' not found")
            return None

        profile = self.profiles[profile_name]

        try:
            module_name = f"hardware.drivers.{self._class_to_module(profile.driver_class)}"
            module = importlib.import_module(module_name)
            driver_class = getattr(module, profile.driver_class)
            driver = driver_class(profile)
            print(f"Created driver: {profile.driver_class} for {profile.name}")
            return driver
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"Driver module not found: {module_name}")
        except AttributeError:
            raise AttributeError(f"Driver class '{profile.driver_class}' not found in module")
        except Exception as e:
            raise RuntimeError(f"Error creating driver: {e}")

    @staticmethod
    def _class_to_module(class_name: str) -> str:
        """
        Convert a CamelCase driver class name to its snake_case module filename.

        The ``Driver`` / ``Device`` suffix is stripped so that e.g.
        ``GenericSerialDriver`` maps to ``generic_serial`` (not ``generic_serial_driver``).

        Examples::

            GenericSerialDriver -> generic_serial
            DebugSerialDriver   -> debug_serial
            TemplateDriver      -> template
            SerialDevice        -> serial_device
        """
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
        if snake.endswith("_driver"):
            snake = snake[:-7]
        return snake

    # ─── Driver accessors ─────────────────────────────────────────────────

    def get_profile_names(self) -> List[str]:
        return sorted(self.profiles.keys())

    def get_profile(self, profile_name: str) -> Optional[HardwareProfile]:
        return self.profiles.get(profile_name)

    def set_current_driver(self, driver: HardwareDriver) -> None:
        self.current_driver = driver

    def get_current_driver(self) -> Optional[HardwareDriver]:
        return self.current_driver

    def save_profile(self, profile: HardwareProfile, filename: Optional[str] = None) -> bool:
        """Save a hardware profile to a JSON file."""
        filename = filename or f"{profile.name.lower().replace(' ', '_')}.json"
        filepath = self.profiles_dir / filename
        try:
            with open(filepath, "w") as f:
                json.dump(
                    {
                        "name": profile.name,
                        "driver": profile.driver_class,
                        "baudrate": profile.baudrate,
                        "timeout": profile.timeout,
                        "terminator": profile.terminator,
                        "return_on_init": profile.return_on_init,
                        "commands": profile.commands,
                        "data_format": profile.data_format,
                    },
                    f,
                    indent=4,
                )
            self.profiles[profile.name] = profile
            return True
        except Exception as e:
            print(f"Error saving profile: {e}")
            return False
