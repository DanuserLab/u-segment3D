function varargout = imageDataGUI(varargin)
%IMAGEDATAGUI MATLAB code file for imageDataGUI.fig
%      IMAGEDATAGUI, by itself, creates a new IMAGEDATAGUI or raises the existing
%      singleton*.
%
%      H = IMAGEDATAGUI returns the handle to a new IMAGEDATAGUI or the handle to
%      the existing singleton*.
%
%      IMAGEDATAGUI('Property','Value',...) creates a new IMAGEDATAGUI using the
%      given property value pairs. Unrecognized properties are passed via
%      varargin to imageDataGUI_OpeningFcn.  This calling syntax produces a
%      warning when there is an existing singleton*.
%
%      IMAGEDATAGUI('CALLBACK') and IMAGEDATAGUI('CALLBACK',hObject,...) call the
%      local function named CALLBACK in IMAGEDATAGUI.M with the given input
%      arguments.
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

% Edit the above text to modify the response to help imageDataGUI

% Last Modified by GUIDE v2.5 23-Jun-2020 14:18:14

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @imageDataGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @imageDataGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [], ...
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


% --- Executes just before imageDataGUI is made visible.
function imageDataGUI_OpeningFcn(hObject, eventdata, handles, varargin)
%
% imageDataGUI('mainFig', handles.figure1) - call from movieSelector
% imageDataGUI(ImD) - ImageData viewer
%
% Useful tools:
%
% User Data:
% 
% userData.imFolders - array of ImFolder objects
% userData.mainFig - handle of movie selector GUI
% userData.handles_main - 'handles' of movie selector GUI
%
% userData.setImFolderFig - handle of imFolder set-up figure % QZ I do not need this now.
% userData.iconHelpFig - handle of help dialog
%
% NOTE: If imageDataGUI is under the "Overview" mode, additionally, 
% 
% userData.ImD - the handle of selected ImageData object
%
%

% Input check
ip = inputParser;
ip.addRequired('hObject',@ishandle);
ip.addRequired('eventdata',@(x) isstruct(x) || isempty(x));
ip.addRequired('handles',@isstruct);
ip.addOptional('ImD',[],@(x) isa(x,'ImageData'));
ip.addParameter('mainFig',-1,@ishandle);
ip.parse(hObject,eventdata,handles,varargin{:})

% Store inpu
userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end
userData.ImD=ip.Results.ImD;
userData.mainFig=ip.Results.mainFig;


set(handles.text_copyright, 'String', getLCCBCopyright());


% Set ImFolder object array
userData.imFolders = ImFolder.empty(1,0);

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


if ~isempty(userData.ImD),
    userData.imFolders = userData.ImD.imFolders_;
    set(handles.listbox_channel, 'String', userData.ImD.getImFolderPaths)
    
    % GUI setting
    set(handles.pushbutton_delete, 'Enable', 'off')
    set(handles.pushbutton_add, 'Enable', 'off')
    set(handles.pushbutton_output, 'Enable', 'off')
    
    set(hObject, 'Name', 'ImageData Detail')
    set(handles.edit_path,'String', userData.ImD.getFullPath)
    set(handles.edit_output, 'String', userData.ImD.outputDirectory_)
    set(handles.edit_notes, 'String', userData.ImD.notes_)
    
    % % GUI setting - parameters
    % propNames={'pixelSize_','pixelSizeZ_','timeInterval_','numAperture_','camBitdepth_'};
    % validProps = ~cellfun(@(x) isempty(userData.ImD.(x)),propNames);
    
    % propNames=propNames(validProps);
    % cellfun(@(x) set(handles.(['edit_' x(1:end-1)]),'Enable','off',...
    %     'String',userData.ImD.(x)),propNames)    
end

% Choose default command line output for imageDataGUI
handles.output = hObject;

% Update handles structure
set(handles.figure1,'UserData',userData)
guidata(hObject, handles);

% UIWAIT makes imageDataGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = imageDataGUI_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function edit_path_Callback(hObject, eventdata, handles)
% hObject    handle to edit_path (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_path as text
%        str2double(get(hObject,'String')) returns contents of edit_path as a double


% --- Executes during object creation, after setting all properties.
function edit_path_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_path (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton_cancel.
function pushbutton_cancel_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_cancel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
delete(handles.figure1);

% --- Executes on button press in pushbutton_done.
function pushbutton_done_Callback(hObject, eventdata, handles)

userData = get(handles.figure1,'UserData');

% Verify imFolders are given
if ~isfield(userData, 'imFolders') || isempty(userData.imFolders)
    errordlg('Please provide at least one imFolder path.',...
        'Empty ImFolder','modal');
    return;    
end

assert(isa(userData.imFolders(1), 'ImFolder'),'User-defined: userData.imFolders are not of class ''ImFolder''') 

% Check output path
outputDir = get(handles.edit_output, 'String');
if isempty(outputDir) || ~exist(outputDir, 'dir')
    errordlg('Please provide a valid output path to save your results.', ...
               'Empty Output Path', 'modal');
    return;    
end

% % Concatenate numerical parameters as movie options
% propNames={'pixelSize_','pixelSizeZ_','timeInterval_','numAperture_','camBitdepth_'};
% propStrings = cell(numel(propNames), 1);
% for i = 1 : numel(propNames)
%     propStrings{i} =get(handles.(['edit_' propNames{i}(1:end-1)]), 'String');
% end

% validProps = ~cellfun(@isempty,propStrings);
% if ~isempty(userData.MD),
%     validProps=validProps & cellfun(@(x)isempty(userData.MD.(x)),propNames');
% end
% propNames=propNames(validProps);
% propValues=num2cell(str2double(propStrings(validProps)))';
 
% movieOptions = vertcat(propNames,propValues);
% movieOptions = reshape(movieOptions,1,numel(propNames)*2);

movieOptions={'notes_', ''};

% If imageDataGUI is under "Overview" mode
if ~isempty(get(handles.edit_notes, 'String'))
    % movieOptions=horzcat(movieOptions,'notes_',get(handles.edit_notes, 'String'));
    movieOptions={'notes_',get(handles.edit_notes, 'String')};
end

if ~isempty(userData.ImD),
    % Overview mode - edit existing ImageData
    if ~isempty(movieOptions)
        try
            set(userData.ImD,movieOptions{:});
        catch ME
            errormsg = sprintf([ME.message '.\n\nImageData edition failed.']);
            errordlg(errormsg, 'User Input Error','modal');
            return;
        end
    end
else
    % Create Image Data
    try
        userData.ImD = ImageData(userData.imFolders, outputDir, movieOptions{:});
    catch ME
        errormsg = sprintf([ME.message '.\n\nImageData creation failed.']);
        errordlg(errormsg, 'User Input Error','modal');
        set(handles.figure1,'UserData',userData)
        return;
    end
end

try
    userData.ImD.sanityCheck;
catch ME
    errormsg = sprintf('%s.\n\nPlease check your ImageData. ImageData is not saved.',ME.message);
    errordlg(errormsg,'ImFolder Error','modal');
    set(handles.figure1,'UserData',userData)
    return;
end

% If new ImageData was created (from movieSelectorGUI)
if ishandle(userData.mainFig), 
    % Retrieve main window userData
    userData_main = get(userData.mainFig, 'UserData');
    
    % Check if files in image list are saved in the same file
    handles_main = guidata(userData.mainFig);
    contentlist = get(handles_main.listbox_movie, 'String');
    imageDataFullPath = userData.ImD.getFullPath;
    if any(strcmp(imageDataFullPath, contentlist))
        errordlg('Cannot overwrite a image data file which is already in the imageData list. Please choose another file name or another path.','Error','modal');
        return
    end
    
    % Append  ImageData object to movie selector panel
    userData_main.ImD = horzcat(userData_main.ImD, userData.ImD);
    set(userData.mainFig, 'UserData', userData_main)
    movieSelectorGUI('refreshDisplay',userData.mainFig,eventdata,guidata(userData.mainFig));
end
% Delete current window
delete(handles.figure1)



% --- Executes on button press in pushbutton_delete.
function pushbutton_delete_Callback(hObject, eventdata, handles)

userData = get(handles.figure1, 'Userdata');

contents = get(handles.listbox_channel,'String');
% Return if list is empty
if isempty(contents), return; end
iImFol = get(handles.listbox_channel,'Value');

% Delete ImFolder object
delete(userData.imFolders(iImFol))
userData.imFolders(iImFol) = [];
contents(iImFol) = [ ];
set(handles.listbox_channel,'String',contents);

% Point 'Value' to the second last item in the list once the 
% last item has been deleted
set(handles.listbox_channel,'Value',max(1,min(iImFol,length(contents))));

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

path = uigetdir(userDir, 'Add ImFolders ...');
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
    hcstoggle = get(handles.checkbox4, 'Value'); % QZ checkbox4 is 'HCS data' checkbox, I do not need it now.
    if hcstoggle == 1
        newImFolder= ImFolder(path, 'hcsPlatestack_', 1);
        if max(size(newImFolder))>1
            for icn = 1:max(size(newImFolder))
                newImFolder(icn).sanityCheck();
            end
        end
        
    else
        newImFolder = ImFolder(path);
        newImFolder.sanityCheck();
    end
catch ME
    errormsg = sprintf('%s.\n\nPlease check this is valid imFolder.',ME.message);
    errordlg(errormsg,'ImFolder Error','modal');
    return
end

% Refresh listbox_channel
userData.imFolders = horzcat(userData.imFolders, newImFolder);

if hcstoggle == 1
    for in = 1:length(userData.imFolders)
        ch_name = strcat(userData.imFolders(in).imFolderPath_, '-', userData.imFolders(in).hcsFlags_.wN);
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



% --- Executes on button press in pushbutton_setting_imFol.
function pushbutton_setting_imFol_Callback(hObject, eventdata, handles)

userData = get(handles.figure1, 'UserData');
if isempty(userData.imFolders), return; end
assert(isa(userData.imFolders(1), 'ImFolder'), 'User-defined: Not a valid ''ImFolder'' object');

userData.setImFolderFig = imFolderGUI('mainFig', handles.figure1, 'modal');

set(handles.figure1,'UserData',userData);


% --- Executes on button press in pushbutton_bfImport.
function pushbutton_bfImport_Callback(hObject, eventdata, handles) % QZ I do not need it now.



% --- Executes on button press in pushbutton_view_metadata.
function pushbutton_view_metadata_Callback(hObject, eventdata, handles) % QZ I do not need it now.



% --- Executes on button press in checkbox_importMetadata.
function checkbox_importMetadata_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_importMetadata (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_importMetadata



function edit_pixelSize_Callback(hObject, eventdata, handles)
% hObject    handle to edit_pixelSize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_pixelSize as text
%        str2double(get(hObject,'String')) returns contents of edit_pixelSize as a double


% --- Executes during object creation, after setting all properties.
function edit_pixelSize_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_pixelSize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_timeInterval_Callback(hObject, eventdata, handles)
% hObject    handle to edit_timeInterval (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_timeInterval as text
%        str2double(get(hObject,'String')) returns contents of edit_timeInterval as a double


% --- Executes during object creation, after setting all properties.
function edit_timeInterval_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_timeInterval (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_numAperture_Callback(hObject, eventdata, handles)
% hObject    handle to edit_numAperture (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_numAperture as text
%        str2double(get(hObject,'String')) returns contents of edit_numAperture as a double


% --- Executes during object creation, after setting all properties.
function edit_numAperture_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_numAperture (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_camBitdepth_Callback(hObject, eventdata, handles)
% hObject    handle to edit_camBitdepth (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_camBitdepth as text
%        str2double(get(hObject,'String')) returns contents of edit_camBitdepth as a double


% --- Executes during object creation, after setting all properties.
function edit_camBitdepth_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_camBitdepth (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
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



function edit_notes_Callback(hObject, eventdata, handles)
% hObject    handle to edit_notes (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_notes as text
%        str2double(get(hObject,'String')) returns contents of edit_notes as a double


% --- Executes during object creation, after setting all properties.
function edit_notes_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_notes (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_output_Callback(hObject, eventdata, handles)
% hObject    handle to edit_output (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_output as text
%        str2double(get(hObject,'String')) returns contents of edit_output as a double


% --- Executes during object creation, after setting all properties.
function edit_output_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_output (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in listbox_channel.
function listbox_channel_Callback(hObject, eventdata, handles)
% hObject    handle to listbox_channel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox_channel contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox_channel


% --- Executes during object creation, after setting all properties.
function listbox_channel_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox_channel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in checkbox4.
function checkbox4_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox4


%%%%%%%% Below functions are in movieDataGUI.m:
% function edit_property_Callback(hObject, eventdata, handles)
% function checkbox4_Callback(hObject, eventdata, handles) % QZ checkbox4 is 'HCS data' checkbox, I do not need it now.