function icon_ButtonDownFcn(hObject, eventdata)
% This function call up a help dialog box when user click any of the icons
% in all GUIs.
%
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
handles = guidata(hObject);

ud = get(hObject, 'UserData');

if ~isempty(ud) && isfield(ud, 'class')

    %% Open url GitHub wiki page, if ud.class belongs to Biosensors package.
    className = isBiosensorsPackRelatedHelpButtons(ud.class);

    if ~isempty(className)

        % Check if there is internet connection.
        try
            response = webread('https://www.google.com/');
        catch ME
            warndlg('No internet connection available. Please connect to the internet to see the documentation.', 'modal');
        end

        url = getBiosensorsPackHelpPgUrl(className);
        web(url)

    else 
    %% Open original txt help file. 
    % Help files are specified by variable "helpFile"
        
        helpFile=[ud.class '.pdf'];
        if exist(helpFile, 'file')
            if ispc || ismac
                open(helpFile)
            elseif isunix
                helpFilePath = which(helpFile);
                system(['evince ' helpFilePath ' &']);
            end
        else
            warndlg(['Cannot find Help file:', helpFile], 'modal');
            return
        end
    end

else


%% Open GUI based help window
    
    userData = get(handles.figure1, 'UserData');
    % Help dialog from MovieData panel
    splitTag = regexp(get(get(hObject,'parent'), 'tag'), '_','split');

    % Pass handle to userData
    % If from package GUI, call pre-defined help dialog
    % if called from setting GUI, call user-defined help dialog 'msgboxGUI'

    if isfield(userData, 'crtProc')
        copyright = getLCCBCopyright();
        % Help dialog from setting panel
        if ~isempty(userData.crtProc)
            userData.helpFig = msgboxGUI('Text', sprintf([get(hObject,'UserData'), ...
                '\n', copyright ]),'Title',['Help - ' userData.crtProc.getName] );
        else
            userData.helpFig = msgboxGUI('Text', sprintf([get(hObject,'UserData'), ...
                '\n', copyright ]),'Title','Help');
        end


    elseif strcmp(splitTag{1}, 'axes') && length(splitTag) >1

            if strcmpi(splitTag{2}, 'help') % Help icon
                if length(splitTag) < 3
                    % Package help
                    userData.packageHelpFig = msgbox(sprintf(get(hObject,'UserData')), ...
                        ['Help - ' userData.crtPackage.getName], 'custom', get(hObject,'CData'), userData.colormap, 'replace');
                else
                    % Process help
                    procID = str2double(splitTag{3});
                    if ~isnan(procID)

                        procName = regexp(userData.crtPackage.getProcessClassNames{procID}, 'Process','split');
                        userData.processHelpFig(procID) = msgbox(sprintf(get(hObject,'UserData')), ...
                         ['Help - ' procName{1}], 'custom', get(hObject,'CData'), userData.colormap, 'replace');
                    end
                end

            else % Process status icon
                procID = str2double(splitTag{3});
                procName =userData.crtPackage.processes_{procID}.getName;
                userData.statusFig = msgbox(get(hObject,'UserData'), ...
                    'Status', 'custom', get(hObject,'CData'), ...
                    userData.colormap, 'replace');            
            end

    else
        userData.iconHelpFig = msgbox(get(hObject,'UserData'), ...
            'Help', 'custom', get(hObject,'CData'), userData.colormap, 'replace'); 
    end

    set(handles.figure1, 'UserData', userData);
end

end

function className = isBiosensorsPackRelatedHelpButtons(inputClass)
% This local fcn is to check if the ud.class is one of 18 classes used in BiosensorsPackage,
% including BiosensorsPackage class, 12 processes, 3 2D-processes in SegmentationProcess, and 2 classes under Tool menu.
% className will return one of 18 class's name, or empty.

% % Below is how to get allRelatedClass by code, 
% % BUT if do so, all files in BiosensorsPackage will be included in other package on GitHub during CI build and deploy stages.
% allRelatedClass = horzcat(BiosensorsPackage.getProcessClassNames, SegmentationProcess.getConcreteClasses');
% helpToolsClass=cell(1,2);
% [helpToolsClass{:}] = deal(func2str(BiosensorsPackage.getTools(1).funHandle), func2str(BiosensorsPackage.getTools(2).funHandle));
% allRelatedClass=horzcat('BiosensorsPackage', allRelatedClass(1:end-1), helpToolsClass);

allRelatedClass = {
    'BiosensorsPackage', ...
    'DarkCurrentCorrectionProcess', ...
    'ShadeCorrectionProcess', ...
    'CropShadeCorrectROIProcess', ...
    'SegmentationProcess', ...
    'BackgroundMasksProcess', ...
    'MaskRefinementProcess', ...
    'BackgroundSubtractionProcess', ...
    'TransformationProcess', ...
    'BleedthroughCorrectionProcess', ...
    'RatioProcess', ...
    'PhotobleachCorrectionProcess', ...
    'OutputRatioProcess', ...
    'ThresholdProcess', ...
    'MultiScaleAutoSegmentationProcess', ...
    'ExternalSegmentationProcess', ...
    'calculateBleedthroughGUI', ...
    'transformCreationGUI'};

outputIdx = ismember(allRelatedClass, inputClass);
if any(outputIdx)
    className = allRelatedClass{outputIdx};
else
    className = [];
end
end

function url = getBiosensorsPackHelpPgUrl(className)
% This local fcn is to get url of the wiki page of Biosensors package.
switch(className)
    case 'BiosensorsPackage'
        url = 'https://github.com/DanuserLab/u-probe/wiki/u-probe-Package-Description';
    case 'DarkCurrentCorrectionProcess'
        url = 'https://github.com/DanuserLab/u-probe/wiki/Dark-Current-Correction-Process-Description';
    case 'ShadeCorrectionProcess'
        url = 'https://github.com/DanuserLab/u-probe/wiki/Shade-Correction-Process-Description';
    case 'CropShadeCorrectROIProcess'
        url = 'https://github.com/DanuserLab/u-probe/wiki/Cropping-Shade-Corrected-Movie-Process-Description';
    case 'SegmentationProcess'
        url = 'https://github.com/DanuserLab/u-probe/wiki/Segmentation-Process-Description';
    case 'BackgroundMasksProcess'
        url = 'https://github.com/DanuserLab/u-probe/wiki/Background-Mask-Process-Description';
    case 'MaskRefinementProcess'
        url = 'https://github.com/DanuserLab/u-probe/wiki/Mask-Refinement-Process-Description';
    case 'BackgroundSubtractionProcess'
        url = 'https://github.com/DanuserLab/u-probe/wiki/Background-Subtraction-Process-Description';
    case 'TransformationProcess'
        url = 'https://github.com/DanuserLab/u-probe/wiki/Transformation-Process-Description';
    case 'BleedthroughCorrectionProcess'
        url = 'https://github.com/DanuserLab/u-probe/wiki/Bleedthrough-Crosstalk-Correction-Process-Description';
    case 'RatioProcess'
        url = 'https://github.com/DanuserLab/u-probe/wiki/Ratioing-Process-Description';
    case 'PhotobleachCorrectionProcess'
        url = 'https://github.com/DanuserLab/u-probe/wiki/Photobleach-Correction-Process-Description';
    case 'OutputRatioProcess'
        url = 'https://github.com/DanuserLab/u-probe/wiki/Ratio-Output-Process-Description';
    case 'ThresholdProcess'
        url = 'https://github.com/DanuserLab/u-probe/wiki/Thresholding-Process-Description';
    case 'MultiScaleAutoSegmentationProcess'
        url = 'https://github.com/DanuserLab/u-probe/wiki/MSA-(multi-scale-automatic)-Segmentation-Process-Description';
    case 'ExternalSegmentationProcess'
        url = 'https://github.com/DanuserLab/u-probe/wiki/External-Segmentation-Process-Description';
    case 'calculateBleedthroughGUI'
        url = 'https://github.com/DanuserLab/u-probe/wiki/Bleedthrough-Coefficient-Calculation-Description';
    case 'transformCreationGUI'
        url = 'https://github.com/DanuserLab/u-probe/wiki/Alignment-Registration-Transform-Creation-Description';
    otherwise
        url = 'https://github.com/DanuserLab/u-probe/wiki';
end
end
