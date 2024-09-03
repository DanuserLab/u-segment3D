function userfcn_lampSwitch(index, value, handles)
% GUI tool function: control the enable/disable value of uicontrols when
% user check/unchecked the checkboxes of processes. The enable/disable
% value of uicontrols depends on package's dependency matrix - dependM
%
% Input:
%   index - the index of current checkbox
%   value - 1: checked   0: unchecked
%   handles - the "handles" of package control panel movie
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

% Chuangang Ren, 08/2010
% Sebastien Besson, Aug 2011

userData = get(handles.figure1, 'UserData');
M = userData.dependM;

% if no follower exists, return.
if ~any(M(:,index)),return; end

% Look at the dependent processes
childProcesses = find(M(:,index)==1);

isProcSuccess = @(x) ~isempty(userData.crtPackage.processes_{x}) && ...
        userData.crtPackage.processes_{x}.success_;

if value
    % Child processes can be enabled if each of their required parents meet
    % one on the following
    % 1 - they are checked
    % 2 - they have been successfully run
    % 3 - they are the current process (redundancy with condition 1???)
    
    isProcChecked = @(x) get(handles.(['checkbox_' num2str(x)]),'Value');
    isCurrentProc = @(x) x==index;
    reqParentProc = @(proc) find(M(proc,:)==1);
    isChildProcValid = @(proc) all(arrayfun(@(x) isProcChecked(x) ||...
        isProcSuccess(x) || isCurrentProc(x),reqParentProc(proc)));
    
    validChildProc = childProcesses(arrayfun(isChildProcValid,childProcesses));
    for childProc = validChildProc'
        % The following code will probably not be executed
        % Leave it here just in case design is changed
        if get(handles.(['checkbox_' num2str(childProc)]),'Value')
            userfcn_lampSwitch(childProc,1,handles)
        else
            % Turn on the childProcesses checkbox
            userfcn_enable(childProc,'on',handles);
        end
    end
    
elseif ~isProcSuccess(index)
    % Process is unchecked and have not been run successfully
    for i =1:length(childProcesses)
        % Turn off and uncheck the follower checkboxes
        userfcn_enable(childProcesses(i),'off',handles,true);
        userfcn_lampSwitch(childProcesses(i),0,handles);
    end   
end

