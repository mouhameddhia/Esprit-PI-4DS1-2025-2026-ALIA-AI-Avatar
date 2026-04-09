import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import './CustomCursor.css';

const CustomCursor = () => {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [isHovering, setIsHovering] = useState(false);

  useEffect(() => {
    const updateMousePosition = (e) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };
    
    // Check if hovering over clickable items
    const handleMouseOver = (e) => {
      if (e.target.tagName.toLowerCase() === 'button' || 
          e.target.tagName.toLowerCase() === 'a' ||
          e.target.closest('button') ||
          e.target.closest('a')) {
        setIsHovering(true);
      } else {
        setIsHovering(false);
      }
    };

    window.addEventListener('mousemove', updateMousePosition);
    window.addEventListener('mouseover', handleMouseOver);

    return () => {
      window.removeEventListener('mousemove', updateMousePosition);
      window.removeEventListener('mouseover', handleMouseOver);
    };
  }, []);

  return (
    <>
      <motion.div 
        className="custom-cursor-dot"
        animate={{ 
          x: mousePosition.x - 4, 
          y: mousePosition.y - 4,
          scale: isHovering ? 0 : 1
        }}
        transition={{ type: 'spring', stiffness: 1000, damping: 28, mass: 0.1 }}
      />
      <motion.div 
        className="custom-cursor-ring"
        animate={{ 
          x: mousePosition.x - 20, 
          y: mousePosition.y - 20,
          scale: isHovering ? 1.5 : 1,
          borderColor: isHovering ? 'rgba(124, 58, 237, 0.8)' : 'rgba(124, 58, 237, 0.4)',
          backgroundColor: isHovering ? 'rgba(124, 58, 237, 0.1)' : 'transparent'
        }}
        transition={{ type: 'spring', stiffness: 250, damping: 20, mass: 0.5 }}
      />
    </>
  );
};

export default CustomCursor;
