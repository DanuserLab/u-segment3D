function varargout = movieDataGUI(varargin)
% MOVIEDATAGUI M-file for movieDataGUI.fig
%      MOVIEDATAGUI, by itself, creates a new MOVIEDATAGUI or raises the existing
%      singleton*.
%
%      H = MOVIEDATAGUI returns the handle to a new MOVIEDATAGUI or the handle to
%      the existing singleton*.
%
%      MOVIEDATAGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MOVIEDATAGUI.M with the given input arguments.
%
%      MOVIEDATAGUI('Property','Value',...) creates a new MOVIEDATAGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before movieDataGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to movieDataGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES
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

% Edit the above text to modify the response to help movieDataGUI

% Last Modified by GUIDE v2.5 20-Jun-2017 22:08:34

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @movieDataGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @movieDataGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before movieDataGUI is made visible.
function movieDataGUI_OpeningFcn(hObject, eventdata, handles, varargin)
%
% movieDataGUI('mainFig', handles.figure1) - call from movieSelector
% movieDataGUI(MD) - MovieData viewer
%
% Useful tools:
%
% User Data:
% 
% userData.channels - array of Channel objects
% userData.mainFig - handle of movie selector GUI
% userData.handles_main - 'handles' of movie selector GUI
%
% userData.setChannelFig - handle of channel set-up figure
% userData.iconHelpFig - handle of help dialog
%
% NOTE: If movieDataGUI is under the "Overview" mode, additionally, 
% 
% userData.MD - the handle of selected MovieData object
%
%

% Input check
ip = inputParser;
ip.addRequired('hObject',@ishandle);
ip.addRequired('eventdata',@(x) isstruct(x) || isempty(x));
ip.addRequired('handles',@isstruct);
ip.addOptional('MD',[],@(x) isa(x,'MovieData'));
ip.addParameter('mainFig',-1,@ishandle);
ip.parse(hObject,eventdata,handles,varargin{:})

% Store inpu
userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end
userData.MD=ip.Results.MD;
userData.mainFig=ip.Results.mainFig;


set(handles.text_copyright, 'String', getLCCBCopyright());


% Set channel object array
userData.channels = Channel.empty(1,0);

% Load help icon from dialogicons.mat
userData = loadLCCBIcons(userData);
supermap(1,:) = get(hObject,'color');

userData.colormap = supermap;

set(handles.figure1,'CurrentAxes',handles.axes_help);
Img = image(userData.questIconData);
set(hObject,'colormap',supermap);
set(gca, 'XLim',get(Img,'XData'),'YLim',get(Img,'YData'),...
    'visible','off');
set(Img,'ButtonDownFcn',@icon_ButtonDownFcn,...
    'UserData', struct('class', mfilename));


if ~isempty(userData.MD),
    userData.channels = userData.MD.channels_;
    set(handles.listbox_channel, 'String', userData.MD.getChannelPaths)
    
    % GUI setting
    set(handles.pushbutton_delete, 'Enable', 'off')
    set(handles.pushbutton_add, 'Enable', 'off')
    set(handles.pushbutton_output, 'Enable', 'off')
    
    set(hObject, 'Name', 'Movie Detail')
    set(handles.edit_path,'String', userData.MD.getFullPath)
    set(handles.edit_output, 'String', userData.MD.outputDirectory_)
    set(handles.edit_notes, 'String', userData.MD.notes_)
    
    % GUI setting - parameters
    propNames={'pixelSize_','pixelSizeZ_','timeInterval_','numAperture_','camBitdepth_'};
    validProps = ~cellfun(@(x) isempty(userData.MD.(x)),propNames);
    
    propNames=propNames(validProps);
    cellfun(@(x) set(handles.(['edit_' x(1:end-1)]),'Enable','off',...
        'String',userData.MD.(x)),propNames)    
end

% Choose default command line output for movieDataGUI
handles.output = hObject;

% Update handles structure
set(handles.figure1,'UserData',userData)
guidata(hObject, handles);

% UIWAIT makes movieDataGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = movieDataGUI_OutputFcn(~, ~, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton_cancel.
function pushbutton_cancel_Callback(~, ~, handles)
% hObject    handle to pushbutton_cancel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
delete(handles.figure1);


% --- Executes on button press in pushbutton_done.
function pushbutton_done_Callback(hObject, eventdata, handles)

userData = get(handles.figure1,'UserData');

% Verify channels are given
if ~isfield(userData, 'channels') || isempty(userData.channels)
    errordlg('Please provide at least one channel path.',...
        'Empty Channel','modal');
    return;    
end

assert(isa(userData.channels(1), 'Channel'),'User-defined: userData.channels are not of class ''Channel''') 

% Check output path
outputDir = get(handles.edit_output, 'String');
if isempty(outputDir) || ~exist(outputDir, 'dir')
    errordlg('Please provide a valid output path to save your results.', ...
               'Empty Output Path', 'modal');
    return;    
end

% Concatenate numerical parameters as movie options
propNames={'pixelSize_','pixelSizeZ_','timeInterval_','numAperture_','camBitdepth_'};
propStrings = cell(numel(propNames), 1);
for i = 1 : numel(propNames)
    propStrings{i} =get(handles.(['edit_' propNames{i}(1:end-1)]), 'String');
end

validProps = ~cellfun(@isempty,propStrings);
if ~isempty(userData.MD),
    validProps=validProps & cellfun(@(x)isempty(userData.MD.(x)),propNames');
end
propNames=propNames(validProps);
propValues=num2cell(str2double(propStrings(validProps)))';
 
movieOptions = vertcat(propNames,propValues);
movieOptions = reshape(movieOptions,1,numel(propNames)*2);

% If movieDataGUI is under "Overview" mode
if ~isempty(get(handles.edit_notes, 'String'))
    movieOptions=horzcat(movieOptions,'notes_',get(handles.edit_notes, 'String'));
end

if ~isempty(userData.MD),
    % Overview mode - edit existing MovieDat
    if ~isempty(movieOptions)
        try
            set(userData.MD,movieOptions{:});
        catch ME
            errormsg = sprintf([ME.message '.\n\nMovie edition failed.']);
            errordlg(errormsg, 'User Input Error','modal');
            return;
        end
    end
else
    % Create Movie Data
    try
        userData.MD = MovieData(userData.channels, outputDir, movieOptions{:});
    catch ME
        errormsg = sprintf([ME.message '.\n\nMovie creation failed.']);
        errordlg(errormsg, 'User Input Error','modal');
        set(handles.figure1,'UserData',userData)
        return;
    end
end

try
    userData.MD.sanityCheck;
catch ME
    errormsg = sprintf('%s.\n\nPlease check your movie data. Movie data is not saved.',ME.message);
    errordlg(errormsg,'Channel Error','modal');
    set(handles.figure1,'UserData',userData)
    return;
end

% If new MovieData was created (from movieSelectorGUI)
if ishandle(userData.mainFig), 
    % Retrieve main window userData
    userData_main = get(userData.mainFig, 'UserData');
    
    % Check if files in movie list are saved in the same file
    handles_main = guidata(userData.mainFig);
    contentlist = get(handles_main.listbox_movie, 'String');
    movieDataFullPath = userData.MD.getFullPath;
    if any(strcmp(movieDataFullPath, contentlist))
        errordlg('Cannot overwrite a movie data file which is already in the movie list. Please choose another file name or another path.','Error','modal');
        return
    end
    
    % Append  MovieData object to movie selector panel
    userData_main.MD = horzcat(userData_main.MD, userData.MD);
    set(userData.mainFig, 'UserData', userData_main)
    movieSelectorGUI('refreshDisplay',userData.mainFig,eventdata,guidata(userData.mainFig));
end
% Delete current window
delete(handles.figure1)


function edit_property_Callback(hObject, eventdata, handles)

set(hObject,'BackgroundColor',[1 1 1])
if isempty(get(hObject,'String')), return; end

propTag = get(hObject,'Tag');
propName = [propTag(length('edit_')+1:end) '_'];
propValue = str2double(get(hObject,'String'));
if ~MovieData.checkValue(propName,propValue)
    warndlg('Invalid property value','Setting Error','modal');
    set(hObject,'BackgroundColor',[1 .8 .8]);
    return
end

% --- Executes on button press in pushbutton_delete.
function pushbutton_delete_Callback(hObject, eventdata, handles)

userData = get(handles.figure1, 'Userdata');

contents = get(handles.listbox_channel,'String');
% Return if list is empty
if isempty(contents), return; end
iChan = get(handles.listbox_channel,'Value');

% Delete channel object
delete(userData.channels(iChan))
userData.channels(iChan) = [];
contents(iChan) = [ ];
set(handles.listbox_channel,'String',contents);

% Point 'Value' to the second last item in the list once the 
% last item has been deleted
set(handles.listbox_channel,'Value',max(1,min(iChan,length(contents))));

set(handles.figure1, 'Userdata', userData)
guidata(hObject, handles);

% --- Executes on button press in pushbutton_add.
function pushbutton_add_Callback(hObject, eventdata, handles)

set(handles.listbox_channel, 'Value', 1)

userData = get(handles.figure1, 'UserData');
if ishandle(userData.mainFig), 
    handles_main = guidata(userData.mainFig);
    userData_main = get(handles_main.figure1, 'UserData');
    userDir =userData_main.userDir;
else
    userDir=pwd;
end

path = uigetdir(userDir, 'Add Channels ...');
if path == 0, return; end

% Get current list
contents = get(handles.listbox_channel,'String');
if any(strcmp(contents,path))
   warndlg('This directory has been selected! Please select a differenct directory.',...
       'Warning','modal');
   return; 
end

% Create path object and save it to userData
try
    hcstoggle = get(handles.checkbox4, 'Value');
    if hcstoggle == 1
        newChannel= Channel(path, 'hcsPlatestack_', 1);
        if max(size(newChannel))>1
            for icn = 1:max(size(newChannel))
                newChannel(icn).sanityCheck();
            end
        end
        
    else
        newChannel = Channel(path);
        newChannel.sanityCheck();
    end
catch ME
    errormsg = sprintf('%s.\n\nPlease check this is valid channel.',ME.message);
    errordlg(errormsg,'Channel Error','modal');
    return
end

% Refresh listbox_channel
userData.channels = horzcat(userData.channels, newChannel);

if hcstoggle == 1
    for in = 1:length(userData.channels)
        ch_name = strcat(userData.channels(in).channelPath_, '-', userData.channels(in).hcsFlags_.wN);
        contents{end+1} = ch_name{1};
    end
else
contents{end+1} = path;
end
set(handles.listbox_channel,'string',contents);

if ishandle(userData.mainFig), 
    userData_main.userDir = fileparts(path);
    set(handles_main.figure1, 'UserData', userData_main)
end

set(handles.figure1, 'Userdata', userData)
guidata(hObject, handles);


% --- Executes during object deletion, before destroying properties.
function figure1_DeleteFcn(hObject, eventdata, handles)

userData = get(handles.figure1, 'UserData');

if isfield(userData, 'iconHelpFig') && ishandle(userData.iconHelpFig)
   delete(userData.iconHelpFig) 
end


% --- Executes on button press in pushbutton_output.
function pushbutton_output_Callback(hObject, eventdata, handles)

pathname = uigetdir(pwd,'Select a directory to store the processes output');
if isnumeric(pathname), return; end

set(handles.edit_output, 'String', pathname);

% --- Executes on button press in pushbutton_setting_chan.
function pushbutton_setting_chan_Callback(hObject, eventdata, handles)

userData = get(handles.figure1, 'UserData');
if isempty(userData.channels), return; end
assert(isa(userData.channels(1), 'Channel'), 'User-defined: Not a valid ''Channel'' object');

userData.setChannelFig = channelGUI('mainFig', handles.figure1, 'modal');

set(handles.figure1,'UserData',userData);

% --- Executes on button press in pushbutton_bfImport.
function pushbutton_bfImport_Callback(hObject, eventdata, handles)

assert(bfCheckJavaPath(), 'Could not load the Bio-Formats library');


outputDir = get(handles.edit_output, 'String');
if isempty(outputDir) || ~exist(outputDir, 'dir')
    errordlg('Please provide a valid output path to save your results.', ...
               'Empty Output Path', 'modal');
    return;    
end

% Note: list of supported formats could be retrieved using
% loci.formats.tools.PrintFormatTable class
[file, path] = uigetfile(bfGetFileExtensions(),...
    'Select image file to import.');
if isequal(file,0) || isequal(path,0), return; end

% Import data into movie using bioformats
loci.common.DebugTools.enableLogging('INFO');
importMetadata = logical(get(handles.checkbox_importMetadata,'Value'));
outputDir = get(handles.edit_output, 'String');
MD = MovieData([path file], importMetadata, 'askUser', true,...
    'outputDirectory', outputDir);

% Update movie selector interface
userData=get(handles.figure1,'UserData');
if ishandle(userData.mainFig), 
    % Append  MovieData object to movie selector panel
    userData_main = get(userData.mainFig, 'UserData');
    userData_main.MD = horzcat(userData_main.MD, MD);
    set(userData.mainFig, 'UserData', userData_main)
    movieSelectorGUI('refreshDisplay',userData.mainFig,eventdata,guidata(userData.mainFig))    
end

% Relaunch this interface in preview mode
movieDataGUI(MD(end));


% --- Executes on button press in checkbox4.
function checkbox4_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%set(handles.checkbox4, 'Value', 1);
% Hint: get(hObject,'Value') returns toggle state of checkbox4


% --- Executes on button press in pushbutton_view_metadata.
function pushbutton_view_metadata_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_view_metadata (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
userData=get(handles.figure1,'UserData');
MD = userData.MD;
try
    MD.getReader().showMetadata();
catch err
    if(~isempty(MD) && isa(MD.getReader(),'BioformatsReader'))
        rethrow(err);
    else
        msgbox('Cannot show metadata for a movie of this type',...
            'Metadata Unavailable');
    end
end



function edit_pixelSizeZ_Callback(hObject, eventdata, handles)
% hObject    handle to edit_pixelSizeZ (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_pixelSizeZ as text
%        str2double(get(hObject,'String')) returns contents of edit_pixelSizeZ as a double


% --- Executes during object creation, after setting all properties.
function edit_pixelSizeZ_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_pixelSizeZ (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
