function out = cliGUI(string,hfig1,procID,varargin)
    % Generic Command Line Interface GUI for arbitrary processes
    %
    % The user is presented with a command line interface via keyboard.
    % The command dbquit will exit debug mode without applying settings.
    % The command dbcont will apply funParams to the process being
    % modified using processGUI_ApplyFcn
    %
    % See also dbcont, dbquit, processGUI_ApplyFcn, noSettingsProcessGUI
%
% Copyright (C) 2024, Danuser Lab - UTSouthwestern 
%
% This file is part of uSegment3D_Package.
% 
% uSegment3D_Package is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% uSegment3D_Package is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with uSegment3D_Package.  If not, see <http://www.gnu.org/licenses/>.
% 
% 
    out = noSettingsProcessGUI(string,hfig1,procID,varargin{:});
    hObject = out;
    eventdata = [];
    handles = guihandles(out);
    set(handles.text34,'String','Check your command line interface');
    
    userData = get(out,'UserData');
    proc = userData.crtProc;
    
    % Display text
%     set(handles.pushbutton_done,'Callback',@(hObject,eventdata) disp('Type <a href="matlab:dbcont">dbcont</a> to apply your settings.'));
%     set(handles.pushbutton_cancel,'Callback',@(hobject,evendata) disp('Type <a href="matlab:dbquit">dbquit</a> to cancel.'));
    set(handles.pushbutton_done,'Callback',@robo_dbcont);
    set(handles.pushbutton_cancel,'Callback',@robo_dbquit);
    disp('---------------------------------------');
    disp('---------------------------------------');
    disp('---------------------------------------');
    disp(' ');
    disp(' ');
    disp('Welcome to Command Line Interface "GUI" <<<<<------------');
    disp(' ');
    disp('---------------------------------------');
    disp('Stored in the struct <a href="matlab:funParams">funParams</a> are the process'' parameters.');
    disp('Type <a href="matlab:openvar(''funParams'')">openvar(''funParams'')</a> to use the variable editor.')
    disp('Type <a href="matlab:dbcont">dbcont</a> to apply your changes to funParam.');
    disp('Type <a href="matlab:dbquit">dbquit</a> to cancel.')
    funParams = userData.crtProc.getParameters();
    who
    funParams
    openvar('funParams');
    
    % Turn over command to the user
    keyboard;
    
    % Set the parameters if we get to this point via dbcont
    processGUI_ApplyFcn(hObject, eventdata, handles,funParams,varargin);
    % Delete the figure if it still exists
    if(ishandle(hObject) && isvalid(hObject))
        delete(hObject);
    end
end
function robo_dbquit(hObject,eventdata)
    disp('Cancelling Command Line GUI. Changes to funParams not saved.');
    parent = get(hObject,'Parent');
    if(ishandle(parent) && isvalid(parent))
        close(parent);
    end
    % From here: https://www.mathworks.com/matlabcentral/answers/281229-calling-dbquit-all-from-the-ui-thread
    % Initialize the java engine 
    import java.awt.*;
    import java.awt.event.*;
    %Create a Robot-object to do the key-pressing
    rob=Robot;
    %Commands for pressing keys:
    % shift + f5 :
    rob.keyPress(KeyEvent.VK_SHIFT)
    rob.keyPress(KeyEvent.VK_F5)
    rob.keyRelease(KeyEvent.VK_SHIFT)
    rob.keyRelease(KeyEvent.VK_F5)
    % end
end
function robo_dbcont(hObject,eventdata)
    disp('Finished Command Line GUI. Changes to funParams saved.');
    % From here: https://www.mathworks.com/matlabcentral/answers/281229-calling-dbquit-all-from-the-ui-thread
    % Initialize the java engine 
    import java.awt.*;
    import java.awt.event.*;
    %Create a Robot-object to do the key-pressing
    rob=Robot;
    %Commands for pressing keys:
    % f5 :
    rob.keyPress(KeyEvent.VK_F5)
    rob.keyRelease(KeyEvent.VK_F5)
    % end
end
