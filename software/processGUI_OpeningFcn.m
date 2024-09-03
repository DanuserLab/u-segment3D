function processGUI_OpeningFcn(hObject, eventdata, handles, string,varargin)
% Common initialization of concrete process GUIs
%
% This function fills various fields of the userData 
%       userData.mainFig - handle to the main figure
%       userData.handles_main - 'handles' of main figure
%       userData.procID - The ID of process in the current package
%       userData.MD - current MovieData array
%       userData.ML - current MovieList array
%       userData.ImD - current ImageData array
%       userData.crtProc - current process
%       userData.crtPackage - current package
%       userData.crtProcClassName - current process class
%       (as defined by the package: can be superclass)
%       userData.procConstr - constructor of current process
%
%       userData.questIconData - help icon image information
%       userData.colormap - color map information
%
% Sebastien Besson May 2011
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

% Add ImageData compatibility
% Updated by Qiongjing (Jenny) Zou, Jun 2020

% Check input
% The mainFig and procID should always be present
% procCOnstr and procName should only be present if the concrete process
% initation is delegated from an abstract class. Else the constructor will
% be directly read from the package constructor list.
ip = inputParser;
ip.addRequired('hObject',@ishandle);
ip.addRequired('eventdata',@(x) isstruct(x) || isempty(x));
ip.addRequired('handles',@isstruct);
ip.addRequired('string',@(x) isequal(x,'mainFig'));
ip.addOptional('mainFig',[],@ishandle);
ip.addOptional('procID',[],@isscalar);
ip.addParameter('procConstr',[],@(x) isa(x,'function_handle'));
ip.addParameter('procClassName','',@ischar);
ip.addParameter('initChannel',0,@isscalar);
ip.parse(hObject,eventdata,handles,string,varargin{:});

% Retrieve userData and read function input 
userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end
userData.mainFig=ip.Results.mainFig;
userData.procID = ip.Results.procID;
userData.procConstr=ip.Results.procConstr;
userData.crtProcClassName = ip.Results.procClassName;
initChannel = ip.Results.initChannel;

% Set up copyright statement
set(handles.text_copyright, 'String', getLCCBCopyright());

% Get current package, movie data and process
userData.handles_main = guidata(userData.mainFig);
userData_main = get(userData.mainFig, 'UserData');
userData.crtPackage = userData_main.crtPackage;
if strcmp(userData.crtPackage.getMovieClass(), 'MovieData')
    userData.MD = userData_main.MD(userData_main.id);
elseif strcmp(userData.crtPackage.getMovieClass(), 'ImageData')
    userData.ImD = userData_main.ImD(userData_main.id);
else
    userData.ML = userData_main.ML(userData_main.id);
end

% If constructor is not inherited from abstract class, read it from package
if isempty(userData.procConstr)
    userData.procConstr = userData.crtPackage.getDefaultProcessConstructors{userData.procID};
    userData.crtProcClassName = userData.crtPackage.getProcessClassNames{userData.procID};
end

% Retrieve crtProc if procID step of the package is set up AND is the same
% class as the current process
try
    crtProcName = userData.crtPackage.processes_{userData.procID}.name_;
catch
    crtProcName = eval([userData.crtProcClassName '.getName']);
end
if isa(userData.crtPackage.processes_{userData.procID},userData.crtProcClassName)    
    userData.crtProc = userData.crtPackage.processes_{userData.procID};
else
    userData.crtProc =[];
end

% Set process names in the text box and figure title
procString = [' Step ' num2str(userData.procID) ': ' crtProcName];
set(handles.text_processName,'String',procString);
figString = [' Setting - ' crtProcName];
set(handles.figure1,'Name',figString);

% Initialize help, preview figure
userData.helpFig=-1;
userData.previewFig=-1;

% Get icon infomation
userData.questIconData = userData_main.questIconData;
userData.colormap = userData_main.colormap;

% If process does not exist, create a default one in user data.
if isempty(userData.crtProc)
    try
        movieClass = userData.crtPackage.getMovieClass();
        if strcmp(movieClass,'MovieData')
            userData.crtProc = userData.procConstr(userData.MD, ...
                userData.crtPackage.outputDirectory_);
        elseif strcmp(movieClass,'ImageData')
            userData.crtProc = userData.procConstr(userData.ImD, ...
                userData.crtPackage.outputDirectory_);
        else
            userData.crtProc = userData.procConstr(userData.ML, ...
                userData.crtPackage.outputDirectory_);
        end
    catch ME
        if ~isequal(ME.identifier,'MATLAB:class:MethodRestricted')
            rethrow(ME);
        end
    end
end

% Check for multiple movies else
if isfield(handles,'checkbox_applytoall')
    if ~isa(userData_main.crtPackage, 'XcorrFluctuationPackage')
        if ~isempty(userData_main.MD) && isempty(userData_main.ImD)
            if numel(userData_main.MD) ==1
                set(handles.checkbox_applytoall,'Value',0,'Visible','off');
            else
                set(handles.checkbox_applytoall, 'Value',...
                    userData_main.applytoall(userData.procID));
            end
        elseif isempty(userData_main.MD) && ~isempty(userData_main.ImD)
            if numel(userData_main.ImD) ==1
                set(handles.checkbox_applytoall,'Value',0,'Visible','off');
            else
                set(handles.checkbox_applytoall, 'Value',...
                    userData_main.applytoall(userData.procID));
            end
        end
    else
        if numel(userData_main.ML) ==1
            set(handles.checkbox_applytoall,'Value',0,'Visible','off');
        else
            set(handles.checkbox_applytoall, 'Value',...
                userData_main.applytoall(userData.procID));
        end
    end
    uicontrol(handles.pushbutton_done);
end

% ----------------------Set up help icon------------------------
namestrcache = get(hObject, 'Name');
if strcmp(namestrcache, ' Setting - Translocation Scoring') ~=1
% Set up help icon
set(hObject,'colormap',userData.colormap);
% Set up package help. Package icon is tagged as '0'
if isfield(handles, 'axes_help')
    set(handles.figure1,'CurrentAxes',handles.axes_help);
    Img = image(userData.questIconData);
    set(gca, 'XLim',get(Img,'XData'),'YLim',get(Img,'YData'),...
        'visible','off','YDir','reverse');
    set(Img,'ButtonDownFcn',@icon_ButtonDownFcn,...
        'UserData', struct('class',userData.crtProcClassName))
end
end
% Update user data and GUI data
set(hObject, 'UserData', userData);
% ----------------------------------------------------------------
if ~initChannel, return; end

funParams = userData.crtProc.funParams_;

% Set up available input channels
if isfield(userData, 'MD')
set(handles.listbox_availableChannels,'String',userData.MD.getChannelPaths(), ...
    'UserData',1:numel(userData.MD.channels_));

channelIndex = funParams.ChannelIndex;
elseif isfield(userData, 'ImD')
set(handles.listbox_availableChannels,'String',userData.ImD.getImFolderPaths(), ...
    'UserData',1:numel(userData.ImD.imFolders_));

channelIndex = funParams.ImFolderIndex;
end
% Find any parent process
parentProc = userData.crtPackage.getParent(userData.procID);
if isempty(userData.crtPackage.processes_{userData.procID}) && ~isempty(parentProc)
    % Check existence of all parent processes
    emptyParentProc = any(cellfun(@isempty,userData.crtPackage.processes_(parentProc)));
    if ~emptyParentProc
        % Intersect channel index with channel index of parent processes
        parentChannelIndex = @(x) userData.crtPackage.processes_{x}.funParams_.ChannelIndex;
        for i = parentProc
            channelIndex = intersect(channelIndex,parentChannelIndex(i));
        end
    end
end

if ~isempty(channelIndex)
    if isfield(userData, 'MD')
    channelString = userData.MD.getChannelPaths(channelIndex);
    elseif isfield(userData, 'ImD')
        channelString = userData.ImD.getImFolderPaths(channelIndex);
    end
else
    channelString = {};
end

set(handles.listbox_selectedChannels,'String',channelString,...
    'UserData',channelIndex);

% Set default channels callback function
set(handles.checkbox_all,'Callback',@(hObject,eventdata)...
    checkallChannels_Callback(hObject,eventdata,guidata(hObject)));
set(handles.pushbutton_select,'Callback',@(hObject,eventdata)...
    selectChannel_Callback(hObject,eventdata,guidata(hObject)));
set(handles.pushbutton_delete,'Callback',@(hObject,eventdata)...
    deleteChannel_Callback(hObject,eventdata,guidata(hObject)));

