function graphFig = graphViewer(mainFig, graphProc, graphProcId, checkedId)
%GRAPHVIEWER creates a graphical interface to display the graph output of a MovieObject
%
% This function creates a list of checkboxes for all graph processes which
% output can be displayed in a standalone figure. It is called by
% movieViewer.
% 
% Input 
%
%   mainFig - the handle of the calling figure.
% 
%   graphProc - The cell array of processes that can be displaye.
%
%   graphProcId - The  array of processes indices that can be displayed.
%
%   checkedId - The  array of processes indices to display at creation.
%
% Output:
%   
%   graphFig - the handle of the graph control interface
%
% See also: graphViewer, movieViewerOptions
%
% Sebastien Besson, Nov 2012
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

% Check existence of viewer
h=findobj(0,'Name','Graphs');
if ~isempty(h), delete(h); end
graphFig=figure('Name','Graphs','Position',[0 0 200 200],...
    'NumberTitle','off','Tag','figure1','Toolbar','none','MenuBar','none',...
    'Color',get(0,'defaultUicontrolBackgroundColor'),'Resize','off',...
    'DeleteFcn', @(h,event) deleteViewer());

userData = get(mainFig, 'UserData');

graphPanel = uipanel(graphFig,'Position',[0 0 1 1],...
    'Title','Graph','BackgroundColor',get(0,'defaultUicontrolBackgroundColor'),...
    'Units','pixels','Tag','uipanel_graph');
hPosition3=10;

% Create series of anonymous function to generate process controls
createProcText= @(panel,i,j,pos,name) uicontrol(panel,'Style','text',...
    'Position',[10 pos 285 20],'Tag',['text_process' num2str(i)],...
    'String',name,'HorizontalAlignment','left','FontWeight','bold');
createOutputText= @(panel,i,j,pos,text) uicontrol(panel,'Style','text',...
    'Position',[25 pos 285 20],'Tag',['text_process' num2str(i) '_output'...
    num2str(j)],'String',text,'HorizontalAlignment','left');
createChannelBox= @(panel,i,j,k,pos,varargin) uicontrol(panel,'Style','checkbox',...
    'Position',[285+30*k pos 20 20],'Tag',['checkbox_process' num2str(i) '_output'...
    num2str(j) '_channel' num2str(k)],varargin{:});
createMovieBox= @(panel,i,j,pos,name,varargin) uicontrol(panel,'Style','checkbox',...
    'Position',[40 pos 200 25],'Tag',['checkbox_process' num2str(i) '_output'...
    num2str(j)],'String',[' ' name],varargin{:});
createInputBox= @(panel,i,j,k,pos,name,varargin) uicontrol(panel,'Style','checkbox',...
    'Position',[40 pos 200 25],'Tag',['checkbox_process' num2str(i) '_output'...
    num2str(j) '_input' num2str(k)],'String',[' ' name],varargin{:});
createInputInputBox= @(panel,i,j,k,l,pos,varargin) uicontrol(panel,'Style','checkbox',...
    'Position',[285+30*l pos 20 20],'Tag',['checkbox_process' num2str(i) '_output'...
    num2str(j) '_input' num2str(k) '_input' num2str(l)],varargin{:});

hPosition3 = createScalarMapOptions(graphPanel,hPosition3);

hPosition3=hPosition3+50;

% Create controls for selecting all other graphs
nProc = numel(graphProc);
for iProc=nProc:-1:1;
    output=graphProc{iProc}.getDrawableOutput;
    if isa(graphProc{iProc},'SignalProcessingProcess');
        input=graphProc{iProc}.getInput;
        nInput=numel(input);
        
        % Create set of boxes for correlation graphs (input/input)
        validOutput = graphProc{iProc}.checkOutput;
        for iOutput=size(validOutput,3):-1:1
            for iInput=nInput:-1:1
                createOutputText(graphPanel,graphProcId(iProc),iInput,hPosition3,input(iInput).name);
                for jInput=1:iInput
                    if validOutput(iInput,jInput,iOutput)
                        createInputInputBox(graphPanel,graphProcId(iProc),iOutput,iInput,jInput,hPosition3,...
                            'Callback',@(h,event) redrawSignalGraph(h,guidata(h)));
                    end
                end
                hPosition3=hPosition3+20;
            end
            createProcText(graphPanel,graphProcId(iProc),iInput,hPosition3,output(iOutput).name);
            hPosition3=hPosition3+20;
        end
        
    else
        % Create boxes for movie -specific graphs
        validOutput = find(strcmp({output.type},'movieGraph'));
        for iOutput=validOutput(end:-1:1)
            createMovieBox(graphPanel,graphProcId(iProc),iOutput,hPosition3,...
                output(iOutput).name,'Callback',@(h,event) redrawGraph(h,guidata(h)));
            hPosition3=hPosition3+20;
        end
        
        % Create boxes for channel-specific graphs
        validOutput = find(strcmp({output.type},'graph'));
        for iOutput=validOutput(end:-1:1)
            validChan = graphProc{iProc}.checkChannelOutput();
            createOutputText(graphPanel,graphProcId(iProc),iOutput,hPosition3,output(iOutput).name);
            arrayfun(@(x) createChannelBox(graphPanel,graphProcId(iProc),iOutput,x,hPosition3,...
                'Callback',@(h,event) redrawGraph(h,guidata(h))),find(validChan));
            hPosition3=hPosition3+20;
        end
        
        % Create boxes for sampled graphs
        validOutput = find(strcmp({output.type},'sampledGraph'));
        for iOutput=validOutput(end:-1:1)
            validChan = graphProc{iProc}.checkChannelOutput();
            createOutputText(graphPanel,graphProcId(iProc),iOutput,hPosition3,output(iOutput).name);
            arrayfun(@(x) createChannelBox(graphPanel,graphProcId(iProc),iOutput,x,hPosition3,...
                'Callback',@(h,event) redrawGraph(h,guidata(h))),find(validChan(iOutput,:)));
            hPosition3=hPosition3+20;
        end
        
        % Create boxes for sampled graphs
        validOutput = find(strcmp({output.type},'signalGraph'));
        for iOutput=validOutput(end:-1:1)
            input=graphProc{iProc}.getInput;
            validInput = find(graphProc{iProc}.checkOutput());
            %                 createOutputText(graphPanel,graphProcId(iProc),iOutput,hPosition3,output(iOutput).name);
            for iInput=fliplr(validInput)
                createInputBox(graphPanel,graphProcId(iProc),iOutput,iInput,hPosition3,...
                    input(iInput).name,'Callback',@(h,event) redrawGraph(h,guidata(h)));
                hPosition3=hPosition3+20;
            end
        end
        
        createProcText(graphPanel,graphProcId(iProc),iOutput,hPosition3,graphProc{iProc}.getName);
        hPosition3=hPosition3+20;
    end
    
end

if ~isempty(graphProc) && isa(userData.MO,'MovieData')
    uicontrol(graphPanel,'Style','text','Position',[160 hPosition3 100 20],...
        'Tag','text_channels','String','Channels');
    arrayfun(@(i) uicontrol(graphPanel,'Style','text',...
        'Position',[285+30*i hPosition3 20 20],...
        'Tag',['text_channel' num2str(i)],'String',i),...
        1:numel(userData.MO.channels_));
end


%% Get image/overlay panel size and resize them
graphPanelSize = getPanelSize(graphPanel);
set(graphPanel,'Position',[10 10  graphPanelSize(1) graphPanelSize(2)])


%% Resize panels and figure
sz=get(0,'ScreenSize');
maxWidth = graphPanelSize(1)+20;
maxHeight = graphPanelSize(2)+20;
set(graphFig,'Position',[3*sz(3)/4 sz(4)/2 maxWidth maxHeight]);

% Update handles structure and attach it to the main figure
handles = guihandles(graphFig);
guidata(handles.figure1, handles);

% Create redraw callbacks
userData.mainFig = mainFig;
set(handles.figure1,'UserData',userData);

%% Set up default parameters
% Auto check input process
for i=checkedId
    h=findobj(graphFig,'-regexp','Tag',['(\w)_process' num2str(i)  '_output1.*'],...
        '-not','Style','text');
    set(h,'Value',1);
end

redrawGraphs(handles)

function hPosition=createScalarMapOptions(graphPanel,hPosition)

uicontrol(graphPanel,'Style','text',...
    'Position',[20 hPosition 200 20],'Tag','text_UpSample',...
    'String',' Upsampling Factor','HorizontalAlignment','left');
uicontrol(graphPanel,'Style','edit','Position',[220 hPosition 50 20],...
    'String','1','BackgroundColor','white','Tag','edit_UpSample',...
    'Callback',@(h,event) redrawGraphs(guidata(h)));

hPosition=hPosition+20;
uicontrol(graphPanel,'Style','text',...
    'Position',[20 hPosition 200 20],'Tag','text_SmoothParam',...
    'String',' Smoothing Parameter','HorizontalAlignment','left');
uicontrol(graphPanel,'Style','edit','Position',[220 hPosition 50 20],...
    'String','.99','BackgroundColor','white','Tag','edit_SmoothParam',...
    'Callback',@(h,event) redrawGraphs(guidata(h)));

hPosition=hPosition+20;
uicontrol(graphPanel,'Style','text',...
    'Position',[10 hPosition 200 20],'Tag','text_scalarMapOptions',...
    'String','Scalar Map options','HorizontalAlignment','left','FontWeight','bold');
  
function size = getPanelSize(hPanel)
if ~ishandle(hPanel), size=[0 0]; return; end
a=get(get(hPanel,'Children'),'Position');
P=vertcat(a{:});
size = [max(P(:,1)+P(:,3))+10 max(P(:,2)+P(:,4))+20];

function redrawGraphs(handles)

graphBoxes = findobj(handles.uipanel_graph,'-regexp','Tag','checkbox_process*');
checkedBoxes = logical(arrayfun(@(x) get(x,'Value'),graphBoxes));
graphTags=arrayfun(@(x) get(x,'Tag'),graphBoxes(checkedBoxes),...
    'UniformOutput',false);
for i=1:numel(graphTags),
    redrawGraph(handles.(graphTags{i}),handles)
end

function redrawGraph(hObject,handles)
graphTag = get(hObject,'Tag');
userData=get(handles.figure1,'UserData');

% Retrieve the id, process nr and channel nr of the selected graphProc
tokens = regexp(graphTag,'^checkbox_process(\d+)_output(\d+)','tokens');
procId=str2double(tokens{1}{1});
outputList = userData.MO.processes_{procId}.getDrawableOutput;
iOutput = str2double(tokens{1}{2});
output = outputList(iOutput).var;

% Discriminate between channel-specific and movie processes
try 
    iFrame = get(findobj(0,'-regexp','Tag','slider_frame'),'Value');
catch
    iFrame = [];
    warning('frame not provided correctly to graphViewer');
end
tokens = regexp(graphTag,'_channel(\d+)$','tokens');
if ~isempty(tokens)
    iChan = str2double(tokens{1}{1});
    figName = [outputList(iOutput).name '_Channel_' num2str(iChan) '_' num2str(iFrame)];
    if strcmp({outputList(iOutput).type},'sampledGraph')
        inputArgs={iChan,iOutput};
    else
        inputArgs={iChan};
    end
else
    tokens = regexp(graphTag,'_input(\d+)$','tokens');
    if ~isempty(tokens)
        iInput = str2double(tokens{1}{1});
        figName = [outputList(iOutput).name ' - ' ...
            userData.MO.processes_{procId}.getInput(iInput).name];
        inputArgs={iInput};
    else
        inputArgs={};
        figName = outputList(iOutput).name;
    end
end

% Remove spaces... otherwise matlab throws error later for struct field
% names
if (any(ismember(figName,' ,.:;!')))
%     warning('Invalid field name: (for struct [MATLAB]), removing spaces');
    figName(ismember(figName,' ,.:;!')) = '_';
end

% Draw or delete the graph figure depending on the checkbox value
h = userData.getFigure(figName);
if ~get(hObject,'Value'),delete(h); return; end
set(h,'Tag','graphFig');

upSample = str2double(get(handles.edit_UpSample,'String'));
smoothParam = str2double(get(handles.edit_SmoothParam,'String'));



if userData.MO.is3D() % && userData.MO.processes_{procId}.is3DP()
    try 
        ZNr = get(findobj(0,'-regexp','Tag','slider_depth'),'Value');
    catch
        ZNr = [];
        warning('Z slice not provided correctly to graphViewer');
    end
    
    userData.MO.processes_{procId}.draw(inputArgs{:}, 'output', output,...
    'UpSample', upSample,'SmoothParam', smoothParam, 'iZ', ZNr,'iFrame', iFrame);
else
    userData.MO.processes_{procId}.draw(inputArgs{:}, 'output', output,...
    'UpSample', upSample,'SmoothParam', smoothParam);
end


set(h,'DeleteFcn',@(h,event)closeGraphFigure(hObject));


function redrawSignalGraph(hObject,handles)
graphTag = get(hObject,'Tag');
userData=get(handles.figure1,'UserData');

% Retrieve the id, process nr and channel nr of the selected graphProc
tokens = regexp(graphTag,'^checkbox_process(\d+)_output(\d+)_input(\d+)_input(\d+)','tokens');
procId=str2double(tokens{1}{1});
iOutput = str2double(tokens{1}{2});
iInput1 = str2double(tokens{1}{3});
iInput2 = str2double(tokens{1}{4});

signalProc = userData.MO.processes_{procId};
figName = signalProc.getOutputTitle(iInput1,iInput2,iOutput);

% Draw or delete the graph figure depending on the checkbox value
h = userData.getFigure(figName);
if ~get(hObject,'Value'),delete(h); return; end
set(h,'Tag','graphFig');

signalProc.draw(iInput1,iInput2,iOutput);
set(h,'DeleteFcn',@(h,event)closeGraphFigure(hObject));

function closeGraphFigure(hObject)
set(hObject,'Value',0);

function deleteViewer()

h = findobj(0,'-regexp','Tag','graphFig');
if ~isempty(h), delete(h); end