-- meltingpot/lua/modules/avatar_tile_dump.lua
local args      = require 'common.args'
local class     = require 'common.class'
local tensor    = require 'system.tensor'
local component = require 'meltingpot.lua.modules.component'
local registry  = require 'meltingpot.lua.modules.component_registry'

--- A per-avatar, egocentric tile‐code observation (e.g. 11×11).
--- Attach this **only** to your avatar prefab!
local AvatarTileDump = class.Class(component.Component)

function AvatarTileDump:__init__(kwargs)
  kwargs = args.parse(kwargs, {
    {'name',   args.default('AvatarTileDump')},
    {'width',  args.numberType},
    {'height', args.numberType},
  })
  AvatarTileDump.Base.__init__(self, kwargs)
  self._fov_w = kwargs.width
  self._fov_h = kwargs.height
end

function AvatarTileDump:getObservationSpec()
  assert(self.gameObject:hasComponent('Avatar'),
         "AvatarTileDump must be attached to an avatar.")
  local idx = self.gameObject:getComponent('Avatar'):getIndex()
  return {
    name  = tostring(idx) .. ".TILE_CODES",
    type  = 'tensor.Int32Tensor',
    shape = { self._fov_w, self._fov_h },
    min   = 0,
    max   = 255,
  }
end

function AvatarTileDump:addObservations(tileSet, _world, observations)
  if not self.gameObject:hasComponent('Avatar') then return end

  local avatar    = self.gameObject:getComponent('Avatar')
  local transform = self.gameObject:getComponent('Transform')
  local sim       = assert(self.gameObject.simulation,
                           "AvatarTileDump: no simulation attached")

  local playerIndex = avatar:getIndex()
  local obs_name    = tostring(playerIndex) .. ".TILE_CODES"

  local fn = function()
    local w, h = self._fov_w, self._fov_h
    local flat = {}
    for i = 1, w*h do flat[i] = 0 end

    -- fetch once per frame
    local all_objs = sim:getAllGameObjects()

    -- Egocentric window parameters
    --local LEFT, RIGHT, FORWARD, BACKWARD = 5, 5, 9, 1
    local LEFT = avatar._config.view.left
    local RIGHT = avatar._config.view.right
    local FORWARD = avatar._config.view.forward
    local BACKWARD = avatar._config.view.backward
    local k = 1

    for dy =  FORWARD, -BACKWARD, -1 do
      for dx = -LEFT, RIGHT do
        -- world coordinates of cell, the -dy is very tricky but verified by video overlay
        local worldPos = transform:getAbsolutePositionFromRelative({dx, - dy})

        -- gather all objects at that worldPos
        local cell_objs = {}
        for _, o in pairs(all_objs) do
          if o:hasComponent('Transform') then
            local p = o:getComponent('Transform'):getPosition()
            -- p might be {x,y} or fields .x/.y
            local ox = p.x or p[1]
            local oy = p.y or p[2]
            if ox == worldPos[1] and oy == worldPos[2] then
              table.insert(cell_objs, o)
            end
          end
        end

        -- pick top priority: Avatar > apple > floorAcorn
        local code = 0
        for _, o in ipairs(cell_objs) do
          if     o:hasComponent('Avatar') and o:getComponent('Avatar'):isAlive() then
            local st = o:getState()
            local idx2 = o:getComponent('Avatar'):getIndex()
            code = 10 + idx2
            if st:find("Bite") or st:find("prepToEat") then
              code = code + 30
            end
            break
          elseif o:getState() == 'apple' then
            code = 1
          elseif o:getState() == 'floorAcorn' then
            code = 2
          end
        end

        flat[k] = code
        k = k + 1
      end
    end

    return tensor.Int32Tensor(flat)
  end

  table.insert(observations, {
    name  = obs_name,
    type  = 'tensor.Int32Tensor',
    shape = { self._fov_w, self._fov_h },
    func  = fn,
  })
end

registry.registerComponent('AvatarTileDump', AvatarTileDump)
return AvatarTileDump
