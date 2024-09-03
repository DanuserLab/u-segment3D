function userfcn_checkAllMovies(procID, value, handles)

% Add ImageData compatibility
% Updated by Qiongjing (Jenny) Zou, July 2020
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

if get(handles.checkbox_all, 'Value')
    
    userData = get(handles.figure1, 'UserData');
    if ~isa(userData.crtPackage, 'XcorrFluctuationPackage')
        if ~isempty(userData.MD) && isempty(userData.ImD)
            n = length(userData.MD);
        elseif isempty(userData.MD) && ~isempty(userData.ImD)
            n = length(userData.ImD);
        end
        for x = setdiff(1:n, userData.id)
            % Recalls the userData that may have been updated by the
            % checkAllMovies function
            userData=get(handles.figure1, 'UserData');
            userData.statusM(x).Checked(procID) = value;
            set(handles.figure1, 'UserData', userData)
            
            dfs_checkAllMovies(procID, value, handles, x)
        end
    else 
        for x = setdiff(1:length(userData.ML), userData.id)
            % Recalls the userData that may have been updated by the
            % checkAllMovies function
            userData=get(handles.figure1, 'UserData');
            userData.statusM(x).Checked(procID) = value;
            set(handles.figure1, 'UserData', userData)
            
            dfs_checkAllMovies(procID, value, handles, x)
        end
    end
end


function dfs_checkAllMovies(procID, value, handles, x)

    userData = get(handles.figure1, 'UserData');
    M = userData.dependM;
    
    if value  % If check

            parentI = find(M(procID, :)==1);
            parentI = parentI(:)';
            
            if isempty(parentI)

                return
            else
                for i = parentI

                    if userData.statusM(x).Checked(i) || ...
                        (~isempty(userData.package(x).processes_{i}) && ...
                                userData.package(x).processes_{i}.success_ )
                        continue 
                    else
                        userData.statusM(x).Checked(i) = 1;
                        set(handles.figure1, 'UserData', userData)
                        dfs_checkAllMovies(i, value, handles, x)
                    end
                end
            end

    else % If uncheck
            
            childProcesses = find(M(:,procID));
            childProcesses = childProcesses(:)';
            
            if isempty(childProcesses) || ...
                (~isempty(userData.package(x).processes_{procID}) ...
                   && userData.package(x).processes_{procID}.success_)
                return;
            else
                for i = childProcesses   
                    if userData.statusM(x).Checked(i)    
                        userData.statusM(x).Checked(i) = 0;
                        set(handles.figure1, 'UserData', userData)
                        dfs_checkAllMovies(i, value, handles, x)                        
                    end
                end
            end        
    end

