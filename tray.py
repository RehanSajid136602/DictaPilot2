"""
DictaPilot system tray module
Provides system tray icon with menu and minimal settings dialog

MIT License
Copyright (c) 2026 Rehan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import sys
import threading
from typing import Optional, Callable

try:
    import pystray
    from pystray import MenuItem as Item
    PYSTRAY_AVAILABLE = True
except ImportError:
    PYSTRAY_AVAILABLE = False

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False


class TrayIcon:
    def __init__(self, status_callback: Optional[Callable[[], str]] = None):
        self.icon: Optional[pystray.Icon] = None
        self.status_callback = status_callback
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
    def _get_icon(self):
        from PIL import Image, ImageDraw
        
        img = Image.new('RGB', (64, 64), color='#2d2d2d')
        draw = ImageDraw.Draw(img)
        
        if self.status_callback:
            status = self.status_callback()
            color = '#4CAF50' if status == 'Idle' else '#FF9800' if status == 'Recording' else '#2196F3'
            draw.ellipse([8, 8, 56, 56], fill=color, outline='#ffffff', width=2)
        else:
            draw.ellipse([8, 8, 56, 56], fill='#4a4a6a', outline='#ffffff', width=2)
        
        return img
    
    def _setup_menu(self):
        menu = (
            Item('DictaPilot', self._show_about),
            Item('Settings', self._open_settings),
            Item('Status: Unknown', self._show_status),
            Item('Quit', self._quit),
        )
        return menu
    
    def _show_about(self, icon, item):
        if TKINTER_AVAILABLE:
            self._show_message('DictaPilot', 'Cross-platform dictation app\nHold F9 to record')
    
    def _show_status(self, icon, item):
        if self.status_callback:
            status = self.status_callback()
            self._show_message('DictaPilot Status', status)
    
    def _open_settings(self, icon, item):
        if TKINTER_AVAILABLE:
            self._show_settings_dialog()
        else:
            self._show_message('Settings', 'Tkinter not available. Configure via environment variables.')
    
    def _show_settings_dialog(self):
        from config import DictaPilotConfig, load_config
        
        root = tk.Tk()
        root.title('DictaPilot Settings')
        root.geometry('400x350')
        root.resizable(False, False)
        
        config = load_config()
        
        frame = ttk.Frame(root, padding='20')
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text='Hotkey:', font=('', 10)).grid(row=0, column=0, sticky='w', pady=5)
        hotkey_var = tk.StringVar(value=config.hotkey)
        ttk.Entry(frame, textvariable=hotkey_var, width=30).grid(row=0, column=1, pady=5, padx=5)
        
        ttk.Label(frame, text='Model:', font=('', 10)).grid(row=1, column=0, sticky='w', pady=5)
        model_var = tk.StringVar(value=config.model)
        model_combo = ttk.Combobox(frame, textvariable=model_var, width=28, 
                                    values=['whisper-large-v3-turbo', 'whisper-1'])
        model_combo.grid(row=1, column=1, pady=5, padx=5)
        
        ttk.Label(frame, text='Smart Mode:', font=('', 10)).grid(row=2, column=0, sticky='w', pady=5)
        smart_mode_var = tk.StringVar(value=config.smart_mode)
        smart_combo = ttk.Combobox(frame, textvariable=smart_mode_var, width=28,
                                   values=['llm', 'heuristic'])
        smart_combo.grid(row=2, column=1, pady=5, padx=5)
        
        ttk.Label(frame, text='Paste Backend:', font=('', 10)).grid(row=3, column=0, sticky='w', pady=5)
        paste_var = tk.StringVar(value=config.paste_backend)
        paste_combo = ttk.Combobox(frame, textvariable=paste_var, width=28,
                                   values=['auto', 'x11', 'pynput', 'keyboard', 'xdotool', 'osascript'])
        paste_combo.grid(row=3, column=1, pady=5, padx=5)
        
        def save_config():
            config.hotkey = hotkey_var.get()
            config.model = model_var.get()
            config.smart_mode = smart_mode_var.get()
            config.paste_backend = paste_var.get()
            config.save()
            root.destroy()
            self._show_message('Settings', 'Configuration saved!')
        
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=20)
        ttk.Button(btn_frame, text='Save', command=save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text='Cancel', command=root.destroy).pack(side=tk.LEFT, padx=5)
        
        root.mainloop()
    
    def _show_message(self, title: str, message: str):
        if TKINTER_AVAILABLE:
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo(title, message)
            root.destroy()
    
    def _quit(self, icon, item):
        self._running = False
        if self.icon:
            self.icon.stop()
    
    def run(self):
        if not PYSTRAY_AVAILABLE:
            print("pystray not installed. Install with: pip install pystray pillow")
            return False
        
        self._running = True
        self.icon = pystray.Icon(
            'dictapilot',
            self._get_icon(),
            'DictaPilot',
            menu=self._setup_menu()
        )
        
        def update_loop():
            import time
            while self._running:
                try:
                    if self.status_callback:
                        status = self.status_callback()
                        self.icon.title = f'DictaPilot - {status}'
                except Exception:
                    pass
                time.sleep(1)
        
        self._thread = threading.Thread(target=update_loop, daemon=True)
        self._thread.start()
        
        self.icon.run()
        return True
    
    def stop(self):
        self._running = False
        if self.icon:
            self.icon.stop()


def run_tray(status_callback: Optional[Callable[[], str]] = None) -> bool:
    """Run the system tray icon"""
    tray = TrayIcon(status_callback)
    return tray.run()


if __name__ == '__main__':
    def dummy_status():
        return 'Idle'
    
    run_tray(dummy_status)
