function status = generateReport(movieException,userData,varargin)
% Generate report from movie exception cell array
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

% Check input
ip = inputParser;
ip.addRequired('movieException', @iscell);
ip.addRequired('userData', @isstruct);
ip.addOptional('reportType','preprocessing',@ischar)
ip.parse(movieException,userData, varargin{:});
reportType = ip.Results.reportType;

% Check exception status
errorMovies = find(~cellfun(@isempty, movieException, 'UniformOutput', true));
status =1;

if isempty(errorMovies), return; end
status = 0;

% Create log message
basicLogMsg = cell(size(movieException));
extendedLogMsg = cell(size(movieException));
for i = errorMovies(:)'
    % Format movie log message
    if ~isempty(userData.MD) && isempty(userData.ImD)
        field = 'MD';
        type = 'Movie'; 
    elseif isempty(userData.MD) && ~isempty(userData.ImD)
        field = 'ImD';
        type = 'ImageData';
    else
        field = 'ML';
        type = 'Movie list'; 
    end
    basicLogMsg{i} = sprintf('%s %d - %s:\n\n', type, i, userData.(field)(i).getFullPath);
    extendedLogMsg{i}=basicLogMsg{i};
    
    % Read exception message and add causes message if any
    for j = 1:length(movieException{i})
        basicLogMsg{i} = [basicLogMsg{i} '--'  ...
            movieException{i}(j).getReport('basic','hyperlinks','off') sprintf('\n')];
        extendedLogMsg{i} = [extendedLogMsg{i} sprintf('-- %s\n\n', movieException{i}(j).message)];
        if ~isempty(movieException{i}(j).cause)
            basicLogMsg{i} = [basicLogMsg{i},sprintf('\nCaused by:\n%s\n',...
                movieException{i}(j).cause{1}.getReport('basic','hyperlinks','off'))];
            extendedLogMsg{i} = [extendedLogMsg{i},...
                movieException{i}(j).cause{1}.getReport('extended','hyperlinks','off')];
        end
    end
    basicLogMsg{i}=sprintf('%s\n',basicLogMsg{i});
    extendedLogMsg{i}=sprintf('%s\n',extendedLogMsg{i});
end

% Add report information
if strcmpi(reportType,'preprocessing'), 
    additionalText=['Please solve the above problems before continuing.'...
        '\n\nThe movie(s) could not be processed.'];
elseif strcmpi(reportType,'postprocessing'), 
    additionalText=...
        ['Please verify your settings are correct. '...
        'Feel free to contact us if you have question regarding this error.'...
        '\n\nPlease help us improve the software by clearly reporting the '...
        'scenario when this error occurs, and the above error information '...
        'to us (error information is also displayed in Matlab command line).'...
        '\n\nFor contact information please refer to the following URL:'...
        '\nhttp://www.utsouthwestern.edu/labs/danuser/software/'];

end

% Display general MATLAB installation information as a header
% Copied from ver.m

% find platform OS
if ispc
   platform = [system_dependent('getos'),' ',system_dependent('getwinsys')];
elseif ismac
    [fail, input] = unix('sw_vers');
    if ~fail
        platform = strrep(input, 'ProductName:', '');
        platform = strrep(platform, sprintf('\t'), '');
        platform = strrep(platform, sprintf('\n'), ' ');
        platform = strrep(platform, 'ProductVersion:', ' Version: ');
        platform = strrep(platform, 'BuildVersion:', 'Build: ');
    else
        platform = system_dependent('getos');
    end
else    
   platform = system_dependent('getos');
end
   
% display platform type
matlabInfo = sprintf(['MATLAB Version %s\nMATLAB License Number: %s\n'...
    'Operating System: %s\nJava VM Version: %s\n'],...
    version,license,platform,char(strread(version('-java'),'%s',1,'delimiter','\n'))); %#ok<REMFF1>

basicReport = [basicLogMsg{:}, sprintf(additionalText)];
extendedReport =[matlabInfo, extendedLogMsg{:}, sprintf(additionalText)];

% Create title
title='The processing of following movie(s)';
if strcmpi(reportType,'preprocessing'), 
    title=[title ' could not be continued:'];
elseif  strcmpi(reportType,'postprocessing'), 
    title=[title ' was terminated by run time error:'];
end

% Check messagebox existence and generate report using msgboxGUI
if isfield(userData, 'msgboxGUI') && ishandle(userData.msgboxGUI)
    delete(userData.msgboxGUI)
end
if isequal(basicReport,extendedReport)
    userData.msgboxGUI = msgboxGUI('title',title,'text', basicReport,...
        'name','Error report');
else
    userData.msgboxGUI = msgboxGUI('title',title,'text', basicReport,...
        'extendedText',extendedReport,'name','Error report');
end
end