const express = require('express');
const router = express.Router();
const pool = require('../db');

/**
 * GET /api/safe-havens
 * Query params:
 *   - lat: latitude
 *   - lon: longitude
 *   - radius: search radius in meters (default: 5000)
 *   - category: embassy, hospital, un, police, hotel, shelter (optional)
 */
router.get('/', async (req, res) => {
  try {
    const { lat, lon, radius = 5000, category } = req.query;

    if (!lat || !lon) {
      return res.status(400).json({
        error: 'Missing required parameters: lat, lon'
      });
    }

    let query = `
      SELECT
        id,
        name,
        category,
        latitude,
        longitude,
        address,
        phone,
        hours,
        verified,
        (
          6371000 * acos(
            cos(radians($1)) * cos(radians(latitude)) *
            cos(radians(longitude) - radians($2)) +
            sin(radians($1)) * sin(radians(latitude))
          )
        ) AS distance
      FROM safe_havens
      WHERE (
        6371000 * acos(
          cos(radians($1)) * cos(radians(latitude)) *
          cos(radians(longitude) - radians($2)) +
          sin(radians($1)) * sin(radians(latitude))
        )
      ) <= $3
    `;

    const params = [parseFloat(lat), parseFloat(lon), parseInt(radius)];

    if (category) {
      query += ` AND category = $4`;
      params.push(category);
    }

    query += ` ORDER BY distance ASC LIMIT 50`;

    const result = await pool.query(query, params);

    res.json({
      success: true,
      count: result.rows.length,
      data: result.rows,
    });
  } catch (error) {
    console.error('Error fetching safe havens:', error);
    res.status(500).json({
      error: 'Failed to fetch safe havens',
      details: error.message
    });
  }
});

/**
 * GET /api/safe-havens/nearest
 * Get the nearest safe haven of any category
 */
router.get('/nearest', async (req, res) => {
  try {
    const { lat, lon } = req.query;

    if (!lat || !lon) {
      return res.status(400).json({
        error: 'Missing required parameters: lat, lon'
      });
    }

    const query = `
      SELECT
        id,
        name,
        category,
        latitude,
        longitude,
        address,
        phone,
        hours,
        verified,
        (
          6371000 * acos(
            cos(radians($1)) * cos(radians(latitude)) *
            cos(radians(longitude) - radians($2)) +
            sin(radians($1)) * sin(radians(latitude))
          )
        ) AS distance
      FROM safe_havens
      ORDER BY distance ASC
      LIMIT 1
    `;

    const result = await pool.query(query, [parseFloat(lat), parseFloat(lon)]);

    if (result.rows.length === 0) {
      return res.status(404).json({
        error: 'No safe havens found'
      });
    }

    res.json({
      success: true,
      data: result.rows[0],
    });
  } catch (error) {
    console.error('Error fetching nearest safe haven:', error);
    res.status(500).json({
      error: 'Failed to fetch nearest safe haven',
      details: error.message
    });
  }
});

module.exports = router;
