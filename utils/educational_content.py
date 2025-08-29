def get_educational_content():
    """Return educational content about space weather"""
    
    content = [
        {
            'title': '**What is Space Weather?**',
            'content': """
            Space weather refers to the environmental conditions in space as influenced by solar activity and the 
            solar wind's interaction with Earth's magnetosphere. Unlike terrestrial weather, space weather involves 
            electromagnetic phenomena that can significantly impact technology and human activities.
            
            **Key Components:**
            **Solar Wind**: A stream of charged particles continuously flowing from the Sun
            **Magnetic Fields**: Complex magnetic field interactions between the Sun and Earth
            **Radiation**: High-energy particles and electromagnetic radiation from solar events
            **Plasma Environment**: Ionized gas in space affecting radio wave propagation
            
            Understanding space weather is crucial for protecting our technology-dependent society from 
            potentially catastrophic disruptions.
            """,
            'image_caption': 'Artist impression of space weather interaction with Earth'
        },
        
        {
            'title': '**Solar Flares: Explosive Solar Events**',
            'content': """
            Solar flares are intense bursts of electromagnetic radiation from the Sun's surface, classified by their 
            X-ray brightness into categories: A, B, C, M, and X (with X being the most powerful).
            
            **Flare Classifications:**
            **C-Class:** Small flares with minimal Earth impact
            **M-Class:** Medium flares causing brief radio blackouts
            **X-Class:** Large flares capable of causing planet-wide radio blackouts and radiation storms
            
            **Impact on Technology:**
            Satellite damage and operational anomalies
            Radio communication disruptions
            GPS navigation errors
            Power grid fluctuations
            Airline route disruptions over polar regions
            
            Solar flares travel at the speed of light, reaching Earth in approximately 8 minutes, making 
            real-time monitoring and rapid response critical for infrastructure protection.
            """,
            'image_caption': 'Solar flare eruption captured by solar observation satellites'
        },
        
        {
            'title': '**Coronal Mass Ejections (CMEs)**',
            'content': """
            Coronal Mass Ejections are massive bursts of solar plasma and magnetic field released from the Sun's 
            corona into space. Unlike solar flares, CMEs involve physical matter traveling through space.
            
            **CME Characteristics:**
            **Speed:** Travel at 300-3000 km/s (1-4 days to reach Earth)
            **Composition:** Billions of tons of plasma and magnetic field
            **Size:** Can be larger than Earth itself
            **Direction:** Earth-directed CMEs pose the greatest threat
            
            **Geomagnetic Storm Effects:**
            Power grid overloads and blackouts
            Satellite orbit decay and system failures
            Enhanced auroral activity
            Pipeline corrosion acceleration
            Compass navigation errors
            
            The 1859 Carrington Event was the most powerful geomagnetic storm in recorded history, caused by 
            a massive CME. A similar event today could cause trillions of dollars in damage to modern infrastructure.
            """,
            'image_caption': 'CME propagation from Sun toward Earth'
        },
        
        {
            'title': '**Geomagnetic Storms and Earth\'s Response**',
            'content': """
            Geomagnetic storms occur when solar wind disturbances interact with Earth's magnetosphere, causing 
            significant variations in the geomagnetic field. These storms are measured using the Kp index (0-9 scale).
            
            **Storm Categories:**
            **G1 (Minor):** Kp=5 - Weak power grid fluctuations
            **G2 (Moderate):** Kp=6 - Spacecraft charging, aurora at high latitudes
            **G3 (Strong):** Kp=7 - Satellite navigation degraded, power systems affected
            **G4 (Severe):** Kp=8 - Widespread charging problems, power grid instability
            **>G5 (Extreme):** Kp=9 - Power grid collapse possible, satellite control problems
            
            **Economic Impact:**
            Geomagnetic storms cause an estimated $10-70 billion annually in economic losses through:
            Power outage costs and infrastructure replacement
            Satellite mission failures and reduced operational life
            Aviation rerouting and communication delays
            Oil pipeline monitoring system disruptions

            
            Modern society's increasing dependence on technology makes geomagnetic storm monitoring and 
            prediction essential for economic stability and public safety.
            """,
            'image_caption': 'Global geomagnetic field disturbance visualization'
        },
        
        {
            'title': '**Satellite Vulnerabilities and Space Weather**',
            'content': """
            Satellites operate in a harsh space environment where they are constantly exposed to charged particles, 
            radiation, and electromagnetic disturbances from space weather events.
            
            **Primary Threats:**
            **Surface Charging:** Accumulation of electric charge causing component failures
            **Deep Dielectric Charging:** Internal charging leading to electronic anomalies
            **Single Event Effects:** High-energy particles causing memory errors or latchup
            **Total Ionizing Dose:** Cumulative radiation damage over satellite lifetime
            **Atmospheric Drag:** Increased atmospheric density during storms affecting orbits
            
            **Operational Impacts:**
            GPS accuracy degradation affecting navigation systems
            Communication satellite outages
            Earth observation mission interruptions
            International Space Station crew safety concerns
            Spacecraft orbit decay and premature re-entry
            
            Protection Strategies:
            Radiation-hardened electronics design
            Autonomous safe mode activation during storms
            Redundant systems and error correction codes
            Strategic orbit adjustments and mission planning
            """,
            'image_caption': 'Satellite in space weather environment showing particle interactions'
        },
        
        {
            'title': '**Power Grid Impacts and Protection**',
            'content': """
            Power grids are particularly vulnerable to space weather due to their extensive conductive networks 
            that can act as antennas for geomagnetic disturbances, inducing dangerous currents.
            
            **Geomagnetically Induced Currents (GICs):**
            Caused by rapid changes in Earth's magnetic field
            Flow through power lines and transformer windings
            Can cause transformer saturation and overheating
            Lead to voltage instability and cascading failures
            
            **Historical Events:**
            **1989 Quebec Blackout:** 6 million people lost power for 9 hours
            **2003 Swedish Power Outage:** Transformer damage in Malmö
            **1989 New Jersey Event:** $6 million transformer destroyed
            
            **Protection Measures:**
            Real-time GIC monitoring systems
            Transformer neutral current blocking devices
            Operational procedures for space weather events
            Strategic reserve management and load shedding
            Improved space weather forecasting integration
            
            A Carrington-level event today could cause $1-2 trillion in damage to the North American power grid 
            alone, with recovery taking 4-10 years due to transformer replacement challenges.
            """,
            'image_caption': 'Power grid network showing GIC flow patterns during geomagnetic storms'
        },
        
        {
            'title': '**Space Weather Monitoring and Prediction**',
            'content': """
            Effective space weather monitoring requires a global network of ground-based and space-based instruments 
            providing real-time data on solar activity and its effects on Earth's environment.
            
            **Monitoring Infrastructure:**
            **Solar Observation:** SOHO, SDO, Parker Solar Probe missions
            **Magnetosphere Monitoring:** ACE, DSCOVR, THEMIS satellites
            **Ground Networks:** Magnetometer chains, riometers, GPS networks
            **Ionospheric Monitoring:** Digisonde networks, incoherent scatter radars
            
            **Prediction Challenges:*
            Complex nonlinear dynamics of the Sun-Earth system
            Limited lead time for accurate forecasting
            Regional variations in space weather effects
            Coupling between different space weather phenomena
            
            **Forecast Products:**
            27-day solar activity outlook
            3-day geomagnetic activity forecast
            Real-time alerts and warnings
            Probabilistic impact assessments

            
            Machine learning and artificial intelligence are revolutionizing space weather prediction, enabling 
            more accurate forecasts and better understanding of complex space weather processes.
            """,
            'image_caption': 'Global space weather monitoring network illustration'
        },
        
        {
            'title': '**Solar Cycle and Long-term Variability**',
            'content': """
            The Sun follows an approximately 11-year cycle of activity, characterized by the number and intensity 
            of sunspots, solar flares, and CMEs. Understanding this cycle is crucial for long-term space weather prediction.
            
            **Solar Cycle Phases:**
            **Solar Minimum:** Few sunspots, lower flare activity, quiet conditions
            **Rising Phase:** Increasing activity, more frequent events
            **Solar Maximum:** Peak activity, highest risk period
            **Declining Phase:** Decreasing activity, but still significant events possible
            
            Current Solar Cycle 25:
            Began in December 2019
            Expected to peak around 2024-2025
            Predicted to be moderate in intensity
            Increased activity expected through 2024
            
            Long-term Trends
            Grand solar minima (like Maunder Minimum 1645-1715)
            Secular variations in magnetic field strength
            Correlation with climate patterns
            Impact on satellite orbital decay rates
            
            While solar maximum periods have higher average activity, extreme events can occur at any phase 
            of the solar cycle, making continuous monitoring essential.
            """,
            'image_caption': 'Solar cycle sunspot number variation over multiple cycles'
        }
    ]
    
    return content

def get_space_weather_glossary():
    """Return glossary of space weather terms"""
    
    glossary = {
        '**Aurora**': 'Natural light displays in polar regions caused by charged particles interacting with Earth\'s atmosphere',
        '**CME**': 'Coronal Mass Ejection - massive burst of solar plasma and magnetic field into space',
        '**Dst Index**': 'Disturbance storm time index measuring geomagnetic storm intensity',
        '**F10.7**': 'Solar radio flux at 10.7 cm wavelength, indicator of solar activity level',
        '**GIC**': 'Geomagnetically Induced Currents - currents induced in conductors by magnetic field changes',
        '**Heliosphere**': 'Region of space dominated by solar wind extending beyond Pluto',
        '**IMF**': 'Interplanetary Magnetic Field - magnetic field carried by solar wind',
        '**Kp Index**': 'Planetary K-index measuring global geomagnetic activity (0-9 scale)',
        '**L-shell**': 'Magnetic shell parameter describing particle drift paths in magnetosphere',
        '**Magnetopause**': 'Boundary between Earth\'s magnetosphere and solar wind',
        '**Parker Spiral**': 'Spiral structure of interplanetary magnetic field due to solar rotation',
        '**Proton Event**': 'Solar energetic particle event with enhanced proton flux',
        '**Reconnection**': 'Process where magnetic field lines break and reconnect, releasing energy',
        '**SEP**': 'Solar Energetic Particles - high-energy particles accelerated by solar events',
        '**Solar Wind**': 'Stream of charged particles continuously flowing from the Sun',
        '**Substorm**': 'Brief disturbance in Earth\'s magnetosphere causing aurora and magnetic variations',
        '**TEC**': 'Total Electron Content - measure of ionospheric electron density affecting GPS',
        '**Van Allen Belts**': 'Radiation belts of trapped particles around Earth'
    }
    
    return glossary

def get_educational_videos():
    """Return list of educational video URLs and descriptions"""
    
    videos = [
        {
            'title': '**Introduction to Space Weather**',
            'description': 'Basic overview of space weather phenomena and their effects on Earth',
            'url': 'https://www.youtube.com/watch?v=oHHSSJDJ4oo',  # NASA video
            'duration': '5:30'
        },
        {
            'title': '**Solar Flares and CMEs Enhanced**',
            'description': 'Detailed explanation of solar eruptive events and their propagation',
            'url': 'https://www.youtube.com/watch?v=HFT7ATLQQx8',  # ESA video
            'duration': '8:15'
        },
        {
            'title': '**Geomagnetic Storms and Power Grids**',
            'description': 'How space weather affects electrical power systems',
            'url': 'https://www.youtube.com/watch?v=7ukQhycKOFw',  # Educational video
            'duration': '6:45'
        }
    ]
    
    return videos

def get_case_studies():
    """Return historical case studies of significant space weather events"""
    
    case_studies = [
        {
            'title': '**The Carrington Event (1859)**',
            'description': """
            The most powerful geomagnetic storm in recorded history occurred on September 1-2, 1859. 
            Telegraph systems worldwide failed, with some operators receiving electric shocks. Aurora 
            were visible as far south as the Caribbean. If a similar event occurred today, it could 
            cause $2-3 trillion in damage and take 4-10 years for complete recovery.
            """,
            'lessons': [
                'Extreme events can exceed all previous observations',
                'Telegraph technology was primitive compared to modern electronics',
                'Global infrastructure vulnerability has increased dramatically',
                'Economic impact would be unprecedented in modern society'
            ]
        },
        {
            'title': '**Quebec Blackout (March 13, 1989)**',
            'description': """
            A geomagnetic storm caused the collapse of Quebec's power grid, leaving 6 million people 
            without electricity for 9 hours. The event demonstrated the vulnerability of power grids 
            to space weather and led to improved monitoring and protection measures.
            """,
            'lessons': [
                'Cascading failures can propagate rapidly through power networks',
                'Transformer damage can take months to repair',
                'Economic losses extend far beyond the power sector',
                'Space weather monitoring became critical for grid operators'
            ]
        },
        {
            'title': '**Halloween Storms (October 2003)**',
            'description': """
            A series of powerful solar flares and CMEs occurred in late October 2003, causing 
            widespread satellite anomalies, airline diversions, and power outages. The events 
            highlighted the global nature of space weather impacts on modern technology.
            """,
            'lessons': [
                'Multiple events can compound impacts',
                'Satellite constellation vulnerabilities became apparent',
                'Aviation industry developed new protocols',
                'International cooperation in monitoring improved'
            ]
        }
    ]
    
    return case_studies
