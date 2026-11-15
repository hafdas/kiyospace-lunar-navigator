"use client";

import React from "react";
import {
  MapContainer,
  Circle,
  CircleMarker,
  Popup,
  Tooltip,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

export interface LunarSiteOption {
  site_id: "site01" | "site04";
  name: string;
  lat_deg: number;
  lon_deg: number;
  ground_elevation_m: number;
}

interface LunarMapProps {
  sites: LunarSiteOption[];
  scenarioASite: "site01" | "site04";
  scenarioBSite: "site01" | "site04";
  activeTarget: "A" | "B";
  onSelectSite: (siteId: "site01" | "site04") => void;
}

/**
 * Converts Lunar South Pole (lat_deg, lon_deg) into planar South Polar Stereographic
 * coordinates [Y, X] in kilometers relative to the South Pole (-90°).
 */
function polarLatLonToXY(latDeg: number, lonDeg: number): [number, number] {
  const rKm = (90.0 + latDeg) * 30.32; // ~30.32 km per degree of latitude on the Moon
  const lonRad = (lonDeg * Math.PI) / 180.0;
  const x = rKm * Math.sin(lonRad);
  const y = rKm * Math.cos(lonRad);
  return [y, x];
}

export default function LunarMap({
  sites,
  scenarioASite,
  scenarioBSite,
  activeTarget,
  onSelectSite,
}: LunarMapProps) {
  const bounds: L.LatLngBoundsExpression = [
    [-25, -25],
    [25, 25],
  ];

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div>
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
            Lunar South Pole 2D Stereographic Navigator (MOON_ME / DE421)
          </h2>
          <p className="text-xs text-slate-400">
            Click a landing site marker to assign it to the active target (
            <span className="text-cyan-400 font-semibold">
              Scenario {activeTarget}
            </span>
            ).
          </p>
        </div>
        <span className="text-[11px] px-2.5 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700">
          Projection: South Polar Stereographic (89.0°S – 90.0°S)
        </span>
      </div>

      <div className="h-72 w-full rounded-lg overflow-hidden border border-slate-800 relative">
        <MapContainer
          crs={L.CRS.Simple}
          center={[0, 0]}
          zoom={3}
          minZoom={2}
          maxZoom={6}
          maxBounds={bounds}
          style={{ height: "100%", width: "100%", background: "#090d16" }}
        >
          {/* Reference Polar Latitude Rings (89.5°S and 89.8°S) */}
          <Circle
            center={[0, 0]}
            radius={15.16}
            pathOptions={{ color: "#334155", weight: 1, dashArray: "4 4", fill: false }}
          />
          <Circle
            center={[0, 0]}
            radius={6.06}
            pathOptions={{ color: "#475569", weight: 1, dashArray: "2 4", fill: false }}
          />

          {/* South Pole 90°S Reference Marker & Shackleton Crater Rim */}
          <CircleMarker
            center={[0, 0]}
            radius={4}
            pathOptions={{ color: "#94a3b8", fillColor: "#f8fafc", fillOpacity: 0.9 }}
          >
            <Tooltip permanent direction="right" offset={[6, 0]}>
              90.0°S Lunar South Pole
            </Tooltip>
          </CircleMarker>

          {/* Interactive Candidate Landing Sites */}
          {sites.map((site) => {
            const pos = polarLatLonToXY(site.lat_deg, site.lon_deg);
            const isSelectedA = site.site_id === scenarioASite;
            const isSelectedB = site.site_id === scenarioBSite;

            let strokeColor = "#64748b";
            if (isSelectedA && isSelectedB) strokeColor = "#a855f7"; // Both A & B
            else if (isSelectedA) strokeColor = "#22d3ee"; // Cyan for A
            else if (isSelectedB) strokeColor = "#f59e0b"; // Amber for B

            return (
              <CircleMarker
                key={site.site_id}
                center={pos}
                radius={10}
                pathOptions={{
                  color: strokeColor,
                  weight: 3,
                  fillColor: strokeColor,
                  fillOpacity: 0.45,
                }}
                eventHandlers={{
                  click: () => onSelectSite(site.site_id),
                }}
              >
                <Tooltip permanent direction="top" offset={[0, -10]}>
                  {site.name} ({site.lat_deg.toFixed(2)}°, {site.lon_deg.toFixed(1)}°)
                </Tooltip>
                <Popup>
                  <div className="text-xs space-y-1 text-slate-900">
                    <p className="font-bold">{site.name}</p>
                    <p>
                      Coordinates: {site.lat_deg.toFixed(4)}° N, {site.lon_deg.toFixed(4)}° E
                    </p>
                    <p>Surface Elevation: {site.ground_elevation_m.toFixed(1)} m</p>
                    <button
                      onClick={() => onSelectSite(site.site_id)}
                      className="mt-1 px-2 py-1 bg-slate-900 text-white rounded text-xs w-full"
                    >
                      Assign to Scenario {activeTarget}
                    </button>
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}
        </MapContainer>
      </div>
    </div>
  );
}